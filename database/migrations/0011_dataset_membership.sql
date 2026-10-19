-- 0011_dataset_membership.sql
--
-- Immutable dataset checkpoints with exact session membership.
-- Dataset metadata uses the existing PGL serialized JSON representation.
-- Scientific payloads are not embedded here.
--
-- Active completed datasets protect their members and all ancestors.
-- Historical action provenance does not independently protect sessions.
--
-- The migration runner supplies the transaction and search_path.

CREATE TABLE dataset (
    dataset_id bigint GENERATED ALWAYS AS IDENTITY
        (START WITH 0 MINVALUE 0) PRIMARY KEY,

    -- Stable identity across versions; names are not global identifiers.
    family_uuid uuid NOT NULL,

    name text NOT NULL CHECK (btrim(name) <> ''),
    checkpoint_name text,
    description text,

    status text NOT NULL DEFAULT 'in_progress',
    contract_version text NOT NULL DEFAULT 'pgl.dataset.v1',

    -- Lightweight serialized metadata, not xarray or session payloads.
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,

    created_by bigint NOT NULL REFERENCES users(user_id),
    created_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    deleted_at timestamptz,

    CONSTRAINT dataset_status_check
        CHECK (status IN ('in_progress', 'complete', 'failed')),

    CONSTRAINT dataset_contract_check
        CHECK (contract_version = 'pgl.dataset.v1'),

    CONSTRAINT dataset_metadata_check
        CHECK (jsonb_typeof(metadata) = 'object'),

    CONSTRAINT dataset_completion_check
        CHECK (status <> 'complete' OR completed_at IS NOT NULL)
);

CREATE INDEX dataset_family_index
ON dataset (family_uuid, dataset_id);

CREATE INDEX dataset_name_index
ON dataset (name);

CREATE INDEX dataset_creator_index
ON dataset (created_by);

CREATE TABLE dataset_members (
    dataset_id bigint NOT NULL REFERENCES dataset(dataset_id),
    session_id bigint NOT NULL REFERENCES session(session_id),

    -- Stable ordering within a checkpoint.
    member_index integer NOT NULL CHECK (member_index >= 0),

    -- Explicit C collation makes uniqueness case-sensitive.
    entry_name text COLLATE "C",
    labels jsonb NOT NULL DEFAULT '{}'::jsonb,

    PRIMARY KEY (dataset_id, session_id),
    UNIQUE (dataset_id, member_index),
    UNIQUE (dataset_id, entry_name),

    CONSTRAINT dataset_entry_name_check
        CHECK (
            entry_name IS NULL
            OR entry_name !~ '^[[:space:]]*$'
        ),

    CONSTRAINT dataset_member_labels_check
        CHECK (jsonb_typeof(labels) = 'object')
);

CREATE INDEX dataset_members_session_index
ON dataset_members (session_id, dataset_id);

-- Serialize dataset writes against session deletion and lineage writes.
--
-- Existing session delete/undelete takes an EXCLUSIVE session table lock.
-- SHARE conflicts with that lock while permitting ordinary reads and
-- other dataset writers holding SHARE.
--
-- Public dataset lifecycle operations must use READ COMMITTED and acquire
-- this lock before performing dataset/payload writes. These statement
-- triggers provide a database-level backstop.
CREATE FUNCTION pgl_lock_dataset_session_dependencies()
RETURNS trigger
LANGUAGE plpgsql
SET search_path FROM CURRENT
AS $$
BEGIN
    IF current_setting('transaction_isolation') <> 'read committed' THEN
        RAISE EXCEPTION
            'Dataset writes require READ COMMITTED isolation';
    END IF;

    LOCK TABLE session IN SHARE MODE;
    RETURN NULL;
END;
$$;

CREATE TRIGGER dataset_lock_session_dependencies
BEFORE INSERT OR UPDATE OR DELETE ON dataset
FOR EACH STATEMENT
EXECUTE FUNCTION pgl_lock_dataset_session_dependencies();

CREATE TRIGGER dataset_members_lock_session_dependencies
BEFORE INSERT OR UPDATE OR DELETE ON dataset_members
FOR EACH STATEMENT
EXECUTE FUNCTION pgl_lock_dataset_session_dependencies();

-- Publication and restoration require every member and ancestor to be
-- complete and active. The session table lock stabilizes this check.
CREATE FUNCTION pgl_require_active_dataset_sources(target_dataset_id bigint)
RETURNS void
LANGUAGE plpgsql
SET search_path FROM CURRENT
AS $$
DECLARE
    blocked_member_id bigint;
    blocked_source_id bigint;
BEGIN
    WITH RECURSIVE ancestry(member_id, source_id) AS (
        SELECT membership.session_id, membership.session_id
        FROM dataset_members AS membership
        WHERE membership.dataset_id = target_dataset_id

        UNION

        SELECT ancestry.member_id, checkpoint.parent_session_id
        FROM ancestry
        JOIN session AS checkpoint
          ON checkpoint.session_id = ancestry.source_id
        WHERE checkpoint.parent_session_id IS NOT NULL
    )
    SELECT ancestry.member_id, ancestry.source_id
    INTO blocked_member_id, blocked_source_id
    FROM ancestry
    LEFT JOIN session AS checkpoint
      ON checkpoint.session_id = ancestry.source_id
    WHERE checkpoint.session_id IS NULL
       OR checkpoint.status <> 'complete'
       OR checkpoint.deleted_at IS NOT NULL
    ORDER BY ancestry.member_id, ancestry.source_id
    LIMIT 1;

    IF FOUND THEN
        RAISE EXCEPTION
            'Dataset % member % requires active completed session %',
            target_dataset_id, blocked_member_id, blocked_source_id;
    END IF;
END;
$$;

-- Published membership is immutable even when the dataset is deleted.
-- Deletion retains membership rows; restoration reuses them.
CREATE FUNCTION pgl_guard_dataset_members()
RETURNS trigger
LANGUAGE plpgsql
SET search_path FROM CURRENT
AS $$
DECLARE
    owner_id bigint;
    owner_status text;
    owner_deleted_at timestamptz;
    source_status text;
    source_deleted_at timestamptz;
BEGIN
    IF TG_OP = 'UPDATE'
       AND NEW.dataset_id IS DISTINCT FROM OLD.dataset_id THEN
        RAISE EXCEPTION
            'Membership cannot be moved between dataset checkpoints';
    END IF;

    IF TG_OP = 'DELETE' THEN
        owner_id := OLD.dataset_id;
    ELSE
        owner_id := NEW.dataset_id;
    END IF;

    -- Serialize membership edits against publication of this owner.
    SELECT status, deleted_at
    INTO owner_status, owner_deleted_at
    FROM dataset
    WHERE dataset_id = owner_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Unknown dataset checkpoint: %', owner_id;
    END IF;

    IF owner_status <> 'in_progress' OR owner_deleted_at IS NOT NULL THEN
        RAISE EXCEPTION
            'Membership can only change on an active in_progress dataset';
    END IF;

    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;

    SELECT status, deleted_at
    INTO source_status, source_deleted_at
    FROM session
    WHERE session_id = NEW.session_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Unknown member session: %', NEW.session_id;
    END IF;

    IF source_status <> 'complete' OR source_deleted_at IS NOT NULL THEN
        RAISE EXCEPTION
            'Dataset member session % must be complete and active',
            NEW.session_id;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER dataset_members_guard
BEFORE INSERT OR UPDATE OR DELETE ON dataset_members
FOR EACH ROW EXECUTE FUNCTION pgl_guard_dataset_members();

CREATE FUNCTION pgl_guard_dataset_checkpoint()
RETURNS trigger
LANGUAGE plpgsql
SET search_path FROM CURRENT
AS $$
DECLARE
    member_count bigint;
    minimum_index integer;
    maximum_index integer;
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.status <> 'in_progress' THEN
            RAISE EXCEPTION
                'A new dataset checkpoint must start in_progress';
        END IF;

        IF NEW.deleted_at IS NOT NULL THEN
            RAISE EXCEPTION
                'A new dataset checkpoint cannot start deleted';
        END IF;

        RETURN NEW;
    END IF;

    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION
            'Dataset checkpoints may only be permanently deleted by garbage collection';
    END IF;

    IF NEW.dataset_id IS DISTINCT FROM OLD.dataset_id
       OR NEW.family_uuid IS DISTINCT FROM OLD.family_uuid THEN
        RAISE EXCEPTION
            'Dataset checkpoint and family identities are immutable';
    END IF;

    IF NEW.deleted_at IS DISTINCT FROM OLD.deleted_at THEN
        IF (to_jsonb(NEW) - 'deleted_at')
            IS DISTINCT FROM
           (to_jsonb(OLD) - 'deleted_at') THEN
            RAISE EXCEPTION
                'Dataset delete/undelete cannot also change metadata';
        END IF;

        IF NEW.deleted_at IS NULL THEN
            PERFORM pgl_require_active_dataset_sources(NEW.dataset_id);
        END IF;

        RETURN NEW;
    END IF;

    IF OLD.deleted_at IS NOT NULL THEN
        RAISE EXCEPTION
            'Deleted dataset % must be undeleted before modification',
            OLD.dataset_id;
    END IF;

    IF OLD.status = 'complete' THEN
        IF (to_jsonb(NEW) - ARRAY['checkpoint_name', 'description']::text[])
            IS DISTINCT FROM
           (to_jsonb(OLD) - ARRAY['checkpoint_name', 'description']::text[]) THEN
            RAISE EXCEPTION
                'Completed dataset % is immutable except for checkpoint_name and description',
                OLD.dataset_id;
        END IF;

        RETURN NEW;
    END IF;

    IF NEW.status = 'complete' THEN
        PERFORM pgl_require_active_dataset_sources(NEW.dataset_id);

        SELECT count(*), min(member_index), max(member_index)
        INTO member_count, minimum_index, maximum_index
        FROM dataset_members
        WHERE dataset_id = NEW.dataset_id;

        -- Empty datasets are allowed. Nonempty membership uses 0..N-1.
        IF member_count > 0
           AND (minimum_index <> 0 OR maximum_index <> member_count - 1) THEN
            RAISE EXCEPTION
                'Dataset membership indices must be consecutive from zero';
        END IF;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER dataset_guard_checkpoint
BEFORE INSERT OR UPDATE OR DELETE ON dataset
FOR EACH ROW EXECUTE FUNCTION pgl_guard_dataset_checkpoint();

-- Protect a member or any of its ancestors from session soft deletion.
--
-- Public lifecycle code will perform a group-wide preflight and translate
-- refusals into pglMessages.warning. This trigger is a final backstop:
-- an exception rolls back the SQL statement rather than partially deleting
-- its protected rows.
CREATE FUNCTION pgl_guard_dataset_session_deletion()
RETURNS trigger
LANGUAGE plpgsql
SET search_path FROM CURRENT
AS $$
DECLARE
    blocking_dataset_id bigint;
    blocking_member_id bigint;
    blocking_entry_name text;
BEGIN
    IF OLD.deleted_at IS NOT NULL OR NEW.deleted_at IS NULL THEN
        RETURN NEW;
    END IF;

    -- Find the target and its descendants. A dataset referencing any of
    -- these protects the target as either its member or an ancestor.
    WITH RECURSIVE descendants(session_id) AS (
        SELECT OLD.session_id

        UNION

        SELECT child.session_id
        FROM session AS child
        JOIN descendants
          ON child.parent_session_id = descendants.session_id
    )
    SELECT owner.dataset_id, membership.session_id, membership.entry_name
    INTO blocking_dataset_id, blocking_member_id, blocking_entry_name
    FROM descendants
    JOIN dataset_members AS membership
      ON membership.session_id = descendants.session_id
    JOIN dataset AS owner
      ON owner.dataset_id = membership.dataset_id
    WHERE owner.status = 'complete'
      AND owner.deleted_at IS NULL
    ORDER BY owner.dataset_id, membership.session_id
    LIMIT 1;

    IF FOUND THEN
        RAISE EXCEPTION
            'Cannot delete session %: active dataset % protects member % (entryName=%) and its ancestors',
            OLD.session_id,
            blocking_dataset_id,
            blocking_member_id,
            COALESCE(blocking_entry_name, '<unset>');
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER session_guard_dataset_dependencies
BEFORE UPDATE OF deleted_at ON session
FOR EACH ROW EXECUTE FUNCTION pgl_guard_dataset_session_deletion();
-- 0007_derived_checkpoint_lineage.sql
-- Permit single-parent derived checkpoints while retaining raw-root rules.
-- Does not create checkpoints, change existing rows, or access payloads.

ALTER TABLE session
    DROP CONSTRAINT session_pilot_origin_check,
    DROP CONSTRAINT session_pilot_kind_check,
    DROP CONSTRAINT session_raw_lineage_check;

ALTER TABLE session
    ADD CONSTRAINT session_origin_check
        CHECK (origin IN ('raw', 'derived')),

    ADD CONSTRAINT session_origin_kind_check
        CHECK (
            (origin = 'raw' AND kind = 'acquired')
            OR
            (origin = 'derived' AND kind = 'processed')
        ),

    ADD CONSTRAINT session_lineage_check
        CHECK (
            (
                origin = 'raw'
                AND parent_session_id IS NULL
                AND root_session_id = session_id
            )
            OR
            (
                origin = 'derived'
                AND parent_session_id IS NOT NULL
                AND parent_session_id <> session_id
                AND root_session_id <> session_id
            )
        );

-- Replace raw-only root assignment with lineage initialization.
-- Keep the old function in place; only its trigger is replaced.
DROP TRIGGER session_set_raw_root ON session;

CREATE FUNCTION pgl_guard_checkpoint_lineage()
RETURNS trigger
LANGUAGE plpgsql
SET search_path FROM CURRENT
AS $$
DECLARE
    parent_root_id bigint;
    parent_status text;
    parent_deleted_at timestamptz;
BEGIN
    IF TG_OP = 'UPDATE' THEN
        -- Lineage cannot change, even while a checkpoint is in progress.
        -- Deletion-marker updates remain permitted by the existing guard.
        IF NEW.origin IS DISTINCT FROM OLD.origin
           OR NEW.kind IS DISTINCT FROM OLD.kind
           OR NEW.parent_session_id IS DISTINCT FROM OLD.parent_session_id
           OR NEW.root_session_id IS DISTINCT FROM OLD.root_session_id THEN
            RAISE EXCEPTION
                'Checkpoint origin, kind, parent, and root are immutable';
        END IF;

        RETURN NEW;
    END IF;

    IF NEW.origin = 'raw' THEN
        IF NEW.parent_session_id IS NOT NULL THEN
            RAISE EXCEPTION 'A raw checkpoint cannot have a parent';
        END IF;

        IF NEW.root_session_id IS NULL THEN
            NEW.root_session_id := NEW.session_id;
        ELSIF NEW.root_session_id IS DISTINCT FROM NEW.session_id THEN
            RAISE EXCEPTION 'A raw checkpoint must be its own root';
        END IF;

        RETURN NEW;
    END IF;

    IF NEW.origin IS DISTINCT FROM 'derived' THEN
        RAISE EXCEPTION 'Unsupported checkpoint origin: %', NEW.origin;
    END IF;

    IF NEW.parent_session_id IS NULL THEN
        RAISE EXCEPTION 'A derived checkpoint requires a parent';
    END IF;

    IF NEW.parent_session_id = NEW.session_id THEN
        RAISE EXCEPTION 'A checkpoint cannot be its own parent';
    END IF;

    -- Hold a lock through the caller's transaction so the parent cannot
    -- be deleted or otherwise updated while the child is being created.
    --
    -- The application lifecycle operations additionally acquire table
    -- locks and recheck descendants before deletion.
    SELECT root_session_id, status, deleted_at
    INTO parent_root_id, parent_status, parent_deleted_at
    FROM session
    WHERE session_id = NEW.parent_session_id
    FOR SHARE;

    IF NOT FOUND THEN
        RAISE EXCEPTION
            'Parent checkpoint % does not exist', NEW.parent_session_id;
    END IF;

    IF parent_status <> 'complete' THEN
        RAISE EXCEPTION
            'Parent checkpoint % must be complete', NEW.parent_session_id;
    END IF;

    IF parent_deleted_at IS NOT NULL THEN
        RAISE EXCEPTION
            'Parent checkpoint % is deleted', NEW.parent_session_id;
    END IF;

    IF NEW.root_session_id IS NULL THEN
        NEW.root_session_id := parent_root_id;
    ELSIF NEW.root_session_id IS DISTINCT FROM parent_root_id THEN
        RAISE EXCEPTION
            'Derived checkpoint root must match its parent root';
    END IF;

    IF NEW.root_session_id = NEW.session_id THEN
        RAISE EXCEPTION 'A derived checkpoint cannot be its own root';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER session_set_lineage
BEFORE INSERT OR UPDATE ON session
FOR EACH ROW EXECUTE FUNCTION pgl_guard_checkpoint_lineage();

-- The existing raw manifest uniqueness index remains unchanged.
-- Derived deduplication policy and indexing are intentionally deferred.
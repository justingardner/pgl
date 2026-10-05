-- 0001_raw_checkpoint_store.sql
-- Compatible subset for the behavioral raw-checkpoint pilot.
-- schema_migrations is created by the migration runner.

CREATE TABLE users (
    user_id bigint GENERATED ALWAYS AS IDENTITY (START WITH 0 MINVALUE 0) PRIMARY KEY,
    username text NOT NULL UNIQUE CHECK (btrim(username) <> ''),
    display_name text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE subjects (
    subject_id text PRIMARY KEY CHECK (btrim(subject_id) <> ''),
    description text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE storage_backends (
    backend_id bigint GENERATED ALWAYS AS IDENTITY (START WITH 0 MINVALUE 0) PRIMARY KEY,
    name text NOT NULL UNIQUE CHECK (btrim(name) <> ''),
    kind text NOT NULL,
    url_prefix text NOT NULL CHECK (btrim(url_prefix) <> ''),
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT storage_backends_kind_check CHECK (kind IN ('nas', 'cloud', 'local'))
);

CREATE TABLE files (
    file_id bigint GENERATED ALWAYS AS IDENTITY (START WITH 0 MINVALUE 0) PRIMARY KEY,
    sha256 char(64) NOT NULL UNIQUE CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    format text,
    kind text NOT NULL DEFAULT 'file',
    created_at timestamptz NOT NULL DEFAULT now(),
    unreferenced_since timestamptz,
    CONSTRAINT files_pilot_kind_check CHECK (kind = 'file')
);

-- Logical names and object keys use relative POSIX-style paths.
-- Reject absolute paths, backslashes, empty segments, "." and "..".
CREATE FUNCTION pgl_valid_relative_path(value text)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
STRICT
AS $$
    SELECT value <> ''
       AND left(value, 1) <> '/'
       AND right(value, 1) <> '/'
       AND strpos(value, chr(92)) = 0
       AND strpos(value, '//') = 0
       AND value !~ '(^|/)[.]{1,2}(/|$)'
$$;

CREATE TABLE storage_locations (
    file_id bigint NOT NULL REFERENCES files(file_id),
    backend_id bigint NOT NULL REFERENCES storage_backends(backend_id),
    object_key text NOT NULL CHECK (pgl_valid_relative_path(object_key)),
    priority integer NOT NULL DEFAULT 0,
    status text NOT NULL DEFAULT 'pending',
    created_at timestamptz NOT NULL DEFAULT now(),
    verified_at timestamptz,
    PRIMARY KEY (file_id, backend_id),
    UNIQUE (backend_id, object_key),
    CONSTRAINT storage_locations_status_check CHECK (status IN ('pending', 'present', 'missing')),
    CONSTRAINT storage_locations_verification_check CHECK (status <> 'present' OR verified_at IS NOT NULL)
);

CREATE TABLE session (
    session_id bigint GENERATED ALWAYS AS IDENTITY (START WITH 0 MINVALUE 0) PRIMARY KEY,
    root_session_id bigint NOT NULL REFERENCES session(session_id),
    parent_session_id bigint REFERENCES session(session_id),
    origin text NOT NULL DEFAULT 'raw',
    kind text NOT NULL DEFAULT 'acquired',
    instrument text,
    ingest_method text,
    acquired_at timestamptz,
    manifest_hash char(64) CHECK (manifest_hash ~ '^[0-9a-f]{64}$'),
    modalities text[] NOT NULL DEFAULT ARRAY['behavior']::text[],
    status text NOT NULL DEFAULT 'in_progress',
    description text,
    meta jsonb,
    contract_version text,
    created_by bigint NOT NULL REFERENCES users(user_id),
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    completed_at timestamptz,

    CONSTRAINT session_pilot_origin_check CHECK (origin = 'raw'),
    CONSTRAINT session_pilot_kind_check CHECK (kind = 'acquired'),
    CONSTRAINT session_raw_lineage_check CHECK (parent_session_id IS NULL AND root_session_id = session_id),
    CONSTRAINT session_ingest_method_check CHECK (ingest_method IN ('manual', 'automated')),
    CONSTRAINT session_status_check CHECK (status IN ('in_progress', 'complete', 'failed')),
    CONSTRAINT session_meta_check CHECK (meta IS NULL OR jsonb_typeof(meta) = 'object'),
    CONSTRAINT session_modalities_check CHECK (cardinality(modalities) > 0 AND array_position(modalities, NULL) IS NULL),
    CONSTRAINT session_completion_check CHECK (status <> 'complete' OR (manifest_hash IS NOT NULL AND completed_at IS NOT NULL)),
    CONSTRAINT session_time_order_check CHECK (completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at)
);

CREATE UNIQUE INDEX session_raw_manifest_unique ON session (manifest_hash) WHERE origin = 'raw';
CREATE INDEX session_root_index ON session (root_session_id);
CREATE INDEX session_parent_index ON session (parent_session_id);
CREATE INDEX session_creator_index ON session (created_by);
CREATE INDEX session_created_index ON session (created_at);
CREATE INDEX session_modalities_index ON session USING gin (modalities);

CREATE TABLE session_files (
    session_id bigint NOT NULL REFERENCES session(session_id),
    name text NOT NULL CHECK (pgl_valid_relative_path(name)),
    file_id bigint NOT NULL REFERENCES files(file_id),
    role text,
    subject_id text REFERENCES subjects(subject_id),
    PRIMARY KEY (session_id, name)
);

CREATE INDEX session_files_file_index ON session_files (file_id);
CREATE INDEX session_files_subject_index ON session_files (subject_id);

-- Fill the raw checkpoint's root from the database-assigned identity.
-- Never assume that the next session ID is zero or contiguous.
CREATE FUNCTION pgl_set_raw_root()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.origin = 'raw' AND NEW.root_session_id IS NULL THEN
        NEW.root_session_id := NEW.session_id;
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER session_set_raw_root
BEFORE INSERT ON session
FOR EACH ROW EXECUTE FUNCTION pgl_set_raw_root();

-- Completed checkpoints cannot be modified or deleted.
-- New checkpoints must begin in_progress so files can be attached first.
CREATE FUNCTION pgl_guard_checkpoint()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.status <> 'in_progress' THEN
            RAISE EXCEPTION 'A new checkpoint must start in_progress';
        END IF;
        RETURN NEW;
    END IF;

    IF OLD.status = 'complete' THEN
        RAISE EXCEPTION 'Completed checkpoint % is immutable', OLD.session_id;
    END IF;

    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;

    IF NEW.session_id IS DISTINCT FROM OLD.session_id THEN
        RAISE EXCEPTION 'Checkpoint IDs cannot be changed';
    END IF;

    IF NEW.status = 'complete' AND NOT EXISTS (
        SELECT 1 FROM session_files WHERE session_id = NEW.session_id
    ) THEN
        RAISE EXCEPTION 'Cannot complete a checkpoint without files';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER session_guard_checkpoint
BEFORE INSERT OR UPDATE OR DELETE ON session
FOR EACH ROW EXECUTE FUNCTION pgl_guard_checkpoint();

-- Lock the owning checkpoint when changing its file links.
-- This serializes file-link edits against checkpoint publication.
CREATE FUNCTION pgl_guard_checkpoint_files()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    checkpoint_id bigint;
    checkpoint_status text;
BEGIN
    IF TG_OP = 'UPDATE' AND NEW.session_id IS DISTINCT FROM OLD.session_id THEN
        RAISE EXCEPTION 'File links cannot be moved between checkpoints';
    END IF;

    IF TG_OP = 'DELETE' THEN
        checkpoint_id := OLD.session_id;
    ELSE
        checkpoint_id := NEW.session_id;
    END IF;

    SELECT status INTO checkpoint_status
    FROM session
    WHERE session_id = checkpoint_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Checkpoint % does not exist', checkpoint_id;
    END IF;

    IF checkpoint_status <> 'in_progress' THEN
        RAISE EXCEPTION 'Files can only be changed on an in_progress checkpoint';
    END IF;

    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER session_files_guard
BEFORE INSERT OR UPDATE OR DELETE ON session_files
FOR EACH ROW EXECUTE FUNCTION pgl_guard_checkpoint_files();

-- Content identity cannot be reassigned to different bytes.
-- Location/verification information belongs in storage_locations.
CREATE FUNCTION pgl_guard_file_identity()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.file_id IS DISTINCT FROM OLD.file_id
       OR NEW.sha256 IS DISTINCT FROM OLD.sha256
       OR NEW.size_bytes IS DISTINCT FROM OLD.size_bytes
       OR NEW.kind IS DISTINCT FROM OLD.kind THEN
        RAISE EXCEPTION 'File content identity is immutable';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER files_guard_identity
BEFORE UPDATE ON files
FOR EACH ROW EXECUTE FUNCTION pgl_guard_file_identity();

CREATE VIEW session_subjects AS
SELECT DISTINCT session_id, subject_id
FROM session_files
WHERE subject_id IS NOT NULL;

CREATE VIEW raw_sessions AS
SELECT *
FROM session
WHERE origin = 'raw';
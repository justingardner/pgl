-- Deletion metadata for session-owned links and shared content.
-- Existing records remain active.

ALTER TABLE session_files
ADD COLUMN deleted_at timestamptz;

ALTER TABLE files
ADD COLUMN deleted_at timestamptz;

ALTER TABLE storage_locations
ADD COLUMN deleted_at timestamptz;

-- Preserve file-link immutability, with a narrow exception:
-- an existing link's deletion marker can follow its owning session.
--
-- Other link metadata remains immutable once the session is complete.
CREATE OR REPLACE FUNCTION pgl_guard_checkpoint_files()
RETURNS trigger
LANGUAGE plpgsql
SET search_path FROM CURRENT
AS $$
DECLARE
    checkpoint_id bigint;
    checkpoint_status text;
    checkpoint_deleted_at timestamptz;
BEGIN
    IF TG_OP = 'UPDATE'
       AND NEW.session_id IS DISTINCT FROM OLD.session_id THEN
        RAISE EXCEPTION 'File links cannot be moved between checkpoints';
    END IF;

    IF TG_OP = 'DELETE' THEN
        checkpoint_id := OLD.session_id;
    ELSE
        checkpoint_id := NEW.session_id;
    END IF;

    SELECT status, deleted_at
    INTO checkpoint_status, checkpoint_deleted_at
    FROM session
    WHERE session_id = checkpoint_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Checkpoint % does not exist', checkpoint_id;
    END IF;

    IF TG_OP = 'INSERT' THEN
        IF NEW.deleted_at IS NOT NULL THEN
            RAISE EXCEPTION 'A new file link cannot start deleted';
        END IF;
    END IF;

    IF TG_OP = 'UPDATE' THEN
        IF NEW.deleted_at IS DISTINCT FROM OLD.deleted_at THEN
            IF (to_jsonb(NEW) - 'deleted_at')
                IS DISTINCT FROM
               (to_jsonb(OLD) - 'deleted_at') THEN
                RAISE EXCEPTION
                    'Delete/undelete cannot also modify file-link metadata';
            END IF;

            IF NEW.deleted_at IS DISTINCT FROM checkpoint_deleted_at THEN
                RAISE EXCEPTION
                    'File-link deletion marker must match its owning checkpoint';
            END IF;

            RETURN NEW;
        END IF;
    END IF;

    IF checkpoint_deleted_at IS NOT NULL THEN
        RAISE EXCEPTION
            'Files cannot be changed on a deleted checkpoint';
    END IF;

    IF checkpoint_status <> 'in_progress' THEN
        RAISE EXCEPTION
            'Files can only be changed on an in_progress checkpoint';
    END IF;

    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;

    RETURN NEW;
END;
$$;
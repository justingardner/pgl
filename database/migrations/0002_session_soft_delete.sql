-- Session-level soft-delete groundwork.
-- NULL means active; a timestamp means soft-deleted.
-- Checkpoint publication status remains unchanged.

ALTER TABLE session
ADD COLUMN deleted_at timestamptz;

-- Preserve checkpoint immutability, allowing only deletion-marker changes.
-- Capture the migration runner's schema-qualified search path.
CREATE OR REPLACE FUNCTION pgl_guard_checkpoint()
RETURNS trigger
LANGUAGE plpgsql
SET search_path FROM CURRENT
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.status <> 'in_progress' THEN
            RAISE EXCEPTION 'A new checkpoint must start in_progress';
        END IF;

        IF NEW.deleted_at IS NOT NULL THEN
            RAISE EXCEPTION 'A new checkpoint cannot start deleted';
        END IF;

        RETURN NEW;
    END IF;

    IF TG_OP = 'DELETE' THEN
        -- No session hard-delete path until garbage collection is implemented.
        RAISE EXCEPTION
            'Checkpoints may only be permanently deleted by garbage collection';
    END IF;

    IF NEW.session_id IS DISTINCT FROM OLD.session_id THEN
        RAISE EXCEPTION 'Checkpoint IDs cannot be changed';
    END IF;

    IF NEW.deleted_at IS DISTINCT FROM OLD.deleted_at THEN
        -- A delete/undelete must not also edit scientific or publication
        -- metadata. Comparing the complete row protects future columns too.
        IF (to_jsonb(NEW) - 'deleted_at')
            IS DISTINCT FROM
           (to_jsonb(OLD) - 'deleted_at') THEN
            RAISE EXCEPTION
                'Delete/undelete cannot also modify checkpoint metadata';
        END IF;

        RETURN NEW;
    END IF;

    IF OLD.deleted_at IS NOT NULL THEN
        RAISE EXCEPTION
            'Deleted checkpoint % must be undeleted before modification',
            OLD.session_id;
    END IF;

    IF OLD.status = 'complete' THEN
        RAISE EXCEPTION
            'Completed checkpoint % is immutable', OLD.session_id;
    END IF;

    IF NEW.status = 'complete' AND NOT EXISTS (
        SELECT 1
        FROM session_files
        WHERE session_id = NEW.session_id
    ) THEN
        RAISE EXCEPTION 'Cannot complete a checkpoint without files';
    END IF;

    RETURN NEW;
END;
$$;

-- File links remain immutable for completed checkpoints.
-- Additionally, deleted in-progress checkpoints cannot acquire, lose,
-- or modify file links.
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
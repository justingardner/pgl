-- Allow description-only edits on active completed checkpoints.
-- Preserve deletion, publication, and scientific-content protections.

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
        RAISE EXCEPTION 'Checkpoints may only be permanently deleted by garbage collection';
    END IF;

    IF NEW.session_id IS DISTINCT FROM OLD.session_id THEN
        RAISE EXCEPTION 'Checkpoint IDs cannot be changed';
    END IF;

    IF NEW.deleted_at IS DISTINCT FROM OLD.deleted_at THEN
        -- Lifecycle changes must remain separate from metadata edits.
        IF (to_jsonb(NEW) - 'deleted_at') IS DISTINCT FROM (to_jsonb(OLD) - 'deleted_at') THEN
            RAISE EXCEPTION 'Delete/undelete cannot also modify checkpoint metadata';
        END IF;

        RETURN NEW;
    END IF;

    IF OLD.deleted_at IS NOT NULL THEN
        RAISE EXCEPTION 'Deleted checkpoint % must be undeleted before modification', OLD.session_id;
    END IF;

    IF OLD.status = 'complete' THEN
        -- Compare the entire row except description, protecting future columns too.
        IF (to_jsonb(NEW) - 'description') IS DISTINCT FROM (to_jsonb(OLD) - 'description') THEN
            RAISE EXCEPTION 'Completed checkpoint % is immutable except for its description', OLD.session_id;
        END IF;

        RETURN NEW;
    END IF;

    IF NEW.status = 'complete' AND NOT EXISTS (
        SELECT 1 FROM session_files WHERE session_id = NEW.session_id
    ) THEN
        RAISE EXCEPTION 'Cannot complete a checkpoint without files';
    END IF;

    RETURN NEW;
END;
$$;

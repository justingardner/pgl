-- 0010_checkpoint_name.sql
-- Add an editable checkpoint label distinct from the longer description.
--
-- Existing descriptions are copied verbatim, including on deleted
-- checkpoints. Descriptions, payload identities, and history are retained.
--
-- The PGL migration runner supplies the surrounding transaction and
-- schema-qualified search_path.

LOCK TABLE session IN ACCESS EXCLUSIVE MODE;

ALTER TABLE session
ADD COLUMN checkpoint_name text;

-- The existing checkpoint guard rejects changes to deleted checkpoints
-- and rejects new metadata fields on completed checkpoints.
--
-- Disable only that guard for this controlled backfill. The table lock
-- prevents concurrent access while the guard is disabled. Both the data
-- changes and trigger state roll back if this migration fails.
ALTER TABLE session DISABLE TRIGGER session_guard_checkpoint;

UPDATE session
SET checkpoint_name = description
WHERE description IS NOT NULL;

ALTER TABLE session ENABLE TRIGGER session_guard_checkpoint;

-- Preserve existing protections while permitting annotation-only edits.
-- Lifecycle marker changes remain separate from annotation changes.
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
        RAISE EXCEPTION
            'Checkpoints may only be permanently deleted by garbage collection';
    END IF;

    IF NEW.session_id IS DISTINCT FROM OLD.session_id THEN
        RAISE EXCEPTION 'Checkpoint IDs cannot be changed';
    END IF;

    IF NEW.deleted_at IS DISTINCT FROM OLD.deleted_at THEN
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
        -- Compare every column except the two editable annotations.
        -- Newly added scientific/publication columns remain protected.
        IF (to_jsonb(NEW) - ARRAY['description', 'checkpoint_name']::text[])
            IS DISTINCT FROM
           (to_jsonb(OLD) - ARRAY['description', 'checkpoint_name']::text[]) THEN
            RAISE EXCEPTION
                'Completed checkpoint % is immutable except for checkpoint_name and description',
                OLD.session_id;
        END IF;

        RETURN NEW;
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

COMMENT ON COLUMN session.checkpoint_name IS
    'Editable human-readable checkpoint label; separate from description and scientific content identity.';
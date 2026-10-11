-- Attaching content to an active checkpoint makes that content active.
-- The existing session_files BEFORE trigger already verifies and locks
-- the owning checkpoint.

CREATE FUNCTION pgl_reactivate_linked_content()
RETURNS trigger
LANGUAGE plpgsql
SET search_path FROM CURRENT
AS $$
BEGIN
    -- Even when already active, UPDATE locks the content row.
    -- Location registration takes the same row lock.
    UPDATE files
    SET deleted_at = NULL
    WHERE file_id = NEW.file_id;

    UPDATE storage_locations
    SET deleted_at = NULL
    WHERE file_id = NEW.file_id
      AND deleted_at IS NOT NULL;

    RETURN NEW;
END;
$$;

CREATE TRIGGER session_files_reactivate_content
AFTER INSERT OR UPDATE OF file_id ON session_files
FOR EACH ROW EXECUTE FUNCTION pgl_reactivate_linked_content();


-- Registering or refreshing a location must not independently restore
-- deleted content. Its marker always follows its content identity.
--
-- Locking the content row serializes this with attachment/reactivation.
CREATE FUNCTION pgl_set_location_deletion_marker()
RETURNS trigger
LANGUAGE plpgsql
SET search_path FROM CURRENT
AS $$
DECLARE
    content_deleted_at timestamptz;
BEGIN
    SELECT deleted_at
    INTO content_deleted_at
    FROM files
    WHERE file_id = NEW.file_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Unknown content identity: %', NEW.file_id;
    END IF;

    NEW.deleted_at := content_deleted_at;
    RETURN NEW;
END;
$$;

CREATE TRIGGER storage_locations_set_deletion_marker
BEFORE INSERT OR UPDATE ON storage_locations
FOR EACH ROW EXECUTE FUNCTION pgl_set_location_deletion_marker();

-- Record the timezone of the client publishing a checkpoint.
-- Leave historical checkpoints NULL: their original saving timezone
-- cannot be recovered from PostgreSQL timestamptz values.

ALTER TABLE session
    ADD COLUMN saved_timezone text;

ALTER TABLE session
    ADD CONSTRAINT session_saved_timezone_nonblank
    CHECK (
        saved_timezone IS NULL
        OR (
            saved_timezone = btrim(saved_timezone)
            AND saved_timezone <> ''
        )
    );

COMMENT ON COLUMN session.saved_timezone IS
    'IANA timezone of the client publishing this checkpoint; NULL means not recorded. '
    'Interpret completed_at in this timezone for local display. '
    'Preserved unchanged when an existing checkpoint is reused.';

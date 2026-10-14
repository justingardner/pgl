SET LOCAL lock_timeout = '10s';
LOCK TABLE session IN ACCESS EXCLUSIVE MODE;
DROP INDEX session_raw_manifest_unique;
CREATE UNIQUE INDEX session_raw_manifest_unique ON session (manifest_hash) WHERE origin = 'raw' AND deleted_at IS NULL;

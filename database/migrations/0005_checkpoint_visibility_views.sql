-- Default session-query views exclude deleted sessions.

CREATE OR REPLACE VIEW raw_sessions AS
SELECT *
FROM session
WHERE origin = 'raw'
  AND deleted_at IS NULL;

CREATE OR REPLACE VIEW session_subjects AS
SELECT DISTINCT link.session_id, link.subject_id
FROM session_files AS link
JOIN session AS checkpoint
  ON checkpoint.session_id = link.session_id
WHERE link.subject_id IS NOT NULL
  AND checkpoint.deleted_at IS NULL
  AND link.deleted_at IS NULL;


-- Explicit inspection views include active and deleted sessions.

CREATE VIEW raw_sessions_including_deleted AS
SELECT *
FROM session
WHERE origin = 'raw';

CREATE VIEW session_subjects_including_deleted AS
SELECT DISTINCT link.session_id, link.subject_id, checkpoint.deleted_at
FROM session_files AS link
JOIN session AS checkpoint
  ON checkpoint.session_id = link.session_id
WHERE link.subject_id IS NOT NULL;
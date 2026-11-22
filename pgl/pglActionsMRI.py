################################################################
#   filename: pglActionsMRI.py
#    purpose: actions for processing MRI data
#         by: JLG
#       date: Nov 15, 2026
################################################################

#############
# Imports
#############
from .pglActionProgress import actionProgress, operationProgress, currentProgress, progressCall, statusOutput, noProgressBar, daskProgress
from copy import deepcopy

from .pglMessages import pglMessages
from .pglPipeline import pglActionCollection

from traitlets import Bool, Dict, Enum, Float, Instance, Int, List, Unicode, Union

from .pglPipeline import pglAction, pglDeferredAction, pglProductSelector


################################################################
# MRI actions
################################################################
class pglActionsMRI(pglActionCollection):
    """MRI preprocessing actions.

    Private preparation helpers do not execute tools, change image geometry,
    publish products, or append action history.
    """

    ################################################################
    # Select acquisitions without imposing geometry relationships
    ################################################################
    @staticmethod
    def _selectAcquisitionScans(mri, scanIndices=None):
        """Return selected scan indices and UUIDs without comparing geometry.

        None selects all attached scans. No implicit scan classification,
        header inspection, or cross-run grouping is performed.
        """
        from .pglMRI import pglMRI

        if not isinstance(mri, pglMRI): raise TypeError("Expected pglMRI.")

        scans = list(mri.scans)
        if not scans: raise ValueError("No MRI scans are attached.")

        if scanIndices is None:
            indices = list(range(len(scans)))
        elif isinstance(scanIndices, (list, tuple)):
            indices = list(scanIndices)
        else:
            raise TypeError("scanIndices must be a list/tuple of indices or None.")

        if not indices: raise ValueError("Select at least one MRI scan.")
        if any(type(index) is not int or not 0 <= index < len(scans) for index in indices):
            raise ValueError("scanIndices must contain valid zero-based session.mri.scans indices.")
        if len(set(indices)) != len(indices):
            raise ValueError("scanIndices must not contain duplicates.")

        identifiers = [scans[index].uuid for index in indices]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Select each acquisition UUID only once.")

        return [{"scanIndex": index, "scanUUID": scans[index].uuid} for index in indices]
    
    ################################################################
    # Prepare geometry groups and explicit run-alignment references
    ################################################################
    @staticmethod
    def _prepareRunAlignment(mri, scanIndices=None, *, referenceRun="middle", allowGeometryMismatch=False, spatialToleranceMM=0.001, orientationTolerance=0.000001):
        """Prepare the inputs for the run-alignment action.

        scanIndices:
            Zero-based indices in session.mri.scans.
            None selects all attached scans; no implicit functional-scan
            classification or acquisition-time sorting is performed.

        referenceRun:
            "first", "middle", or "last", independently within each group.
            An integer is an explicit index in session.mri.scans.
            A list provides one choice per geometry group.
            Middle means len(group) // 2, the later middle for even counts.

        Geometry:
            Uses preserved acquisition-header metadata, or reads the source
            header when no original snapshot exists.
            Never reads editable metadata or accesses scan.data.

            Requires a coded qform and recognized spatial units.
            Uses qform for acquisition-grid comparisons; sform does not
            participate and may describe a different coordinate system.
            A coded qform alone is not proof that its coordinates faithfully
            represent the physical acquisition.

            Compares spatial shape, voxel sizes, orientation, origin, and
            voxel-center corner positions. Unit conversion is for comparison
            only and never changes the image/header.

            Tolerances are absolute. spatialToleranceMM is in millimeters;
            orientationTolerance applies to normalized axis components.

        allowGeometryMismatch:
            False forms independent geometry-compatible alignment groups.
            True requires one explicit selected reference index and authorizes
            one cross-geometry alignment group. Application must then resolve
            its output grid explicitly if the selected geometries differ.

        Returns:
            A JSON-compatible resolved plan for the action to record.

        Errors raise internally to the action runner. Multiple independent
        groups, or an authorized mismatch, warn at level=1 and continue.

        Does not estimate transforms, select reference volumes, read voxel
        arrays, or mutate scans. Backing filesystems may independently verify
        or spool source files when their header streams are opened.

        Concurrent modification of scans, sources, or metadata is unsupported.
        """
        import math
        import numpy as np

        from .pglMRI import pglMRI, pglMRINiftiMetadata

        if not isinstance(mri, pglMRI): raise TypeError("Expected pglMRI.")
        if type(allowGeometryMismatch) is not bool:
            raise TypeError("allowGeometryMismatch must be Boolean.")

        for name, value in (
            ("spatialToleranceMM", spatialToleranceMM),
            ("orientationTolerance", orientationTolerance),
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")

        selectedScans = pglActionsMRI._selectAcquisitionScans(mri, scanIndices)
        scans = list(mri.scans)
        indices = [record["scanIndex"] for record in selectedScans]

        def validChoice(value):
            return type(value) is int or (
                isinstance(value, str) and value in ("first", "middle", "last")
            )

        if isinstance(referenceRun, list):
            if not referenceRun or not all(validChoice(value) for value in referenceRun):
                raise ValueError("Invalid per-group referenceRun choices.")
        elif not validChoice(referenceRun):
            raise ValueError("referenceRun must be an index, first/middle/last, or a list of those choices.")

        if allowGeometryMismatch and type(referenceRun) is not int:
            raise ValueError("Cross-geometry alignment requires an explicit referenceRun index.")

        geometries = {}
        scanRecords = []

        for index in indices:
            scan = scans[index]
            snapshot = scan.sourceMetadata.get("nifti")

            # Acquisition geometry remains authoritative even when a working
            # image or editable metadata contains different header values.
            if snapshot is None:
                if scan._backing is not None:
                    snapshot = pglMRINiftiMetadata.readBacking(scan._backing)
                elif scan.localPath:
                    snapshot = pglMRINiftiMetadata.read(scan.localPath)
                else:
                    raise ValueError(f"Scan {index} has no acquisition header or source.")

            if not isinstance(snapshot, dict):
                raise ValueError(f"Scan {index}: invalid acquisition-header snapshot.")

            fields = snapshot.get("fields")
            transforms = snapshot.get("transforms")
            if not isinstance(fields, dict) or not isinstance(transforms, dict):
                raise ValueError(f"Scan {index}: incomplete acquisition-header snapshot.")

            dimensions = fields.get("dim")
            if (
                not isinstance(dimensions, list) or not dimensions
                or type(dimensions[0]) is not int or dimensions[0] not in (3, 4)
                or len(dimensions) <= dimensions[0]
                or any(type(size) is not int or size <= 0 for size in dimensions[1:dimensions[0] + 1])
            ):
                raise ValueError(f"Scan {index}: select a valid 3D/4D image; higher axes are not implicitly collapsed.")

            shape = dimensions[1:4]
            volumeCount = dimensions[4] if dimensions[0] == 4 else 1
            units = fields.get("xyzt_units")
            factor = {1: 1000.0, 2: 1.0, 3: 0.001}.get(units & 7) if type(units) is int else None
            if factor is None:
                raise ValueError(f"Scan {index}: unknown or unsupported spatial units.")

            qformCode = fields.get("qform_code")
            sformCode = fields.get("sform_code")
            if type(qformCode) is not int or qformCode <= 0:
                raise ValueError(f"Scan {index}: a coded acquisition qform is required; geometry will not be guessed.")
            if transforms.get("qform_code", qformCode) != qformCode:
                raise ValueError(f"Scan {index}: inconsistent recorded qform code.")

            def readAffine(value, label):
                """Interpret a recorded affine in mm without editing its source."""
                try:
                    matrix = np.asarray(value, dtype=float)
                except (TypeError, ValueError):
                    raise ValueError(f"Scan {index}: invalid {label}.") from None

                if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                    raise ValueError(f"Scan {index}: invalid or nonfinite {label}.")
                if not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-8):
                    raise ValueError(f"Scan {index}: invalid {label} homogeneous row.")

                matrix = matrix.copy()
                matrix[:3] *= factor
                if np.linalg.matrix_rank(matrix[:3, :3]) != 3:
                    raise ValueError(f"Scan {index}: singular {label}.")
                return matrix

            qform = readAffine(transforms.get("qform"), "qform")
            sizes = np.linalg.norm(qform[:3, :3], axis=0)
            directions = qform[:3, :3] / sizes
            origin = qform[:3, 3]
            corners = np.array([
                [x, y, z, 1.0]
                for x in (0, shape[0] - 1)
                for y in (0, shape[1] - 1)
                for z in (0, shape[2] - 1)
            ]).T

            pixelDimensions = fields.get("pixdim")
            if not isinstance(pixelDimensions, list) or len(pixelDimensions) < 4:
                raise ValueError(f"Scan {index}: missing voxel sizes.")

            spacing = np.asarray(pixelDimensions[1:4], dtype=float) * factor
            if (
                not np.isfinite(spacing).all() or np.any(spacing <= 0)
                or not np.allclose(spacing, sizes, rtol=0, atol=spatialToleranceMM)
            ):
                raise ValueError(f"Scan {index}: pixdim and qform voxel sizes disagree.")

            # Group acquisition prescriptions using qform, not sform.
            # The sform may independently describe an aligned coordinate system.
            geometries[index] = {
                "shape": shape,
                "matrix": qform,
                "sizes": sizes,
                "directions": directions,
                "origin": origin,
                "corners": corners,
            }
            scanRecords.append({
                "scanIndex": index,
                "scanUUID": scan.uuid,
                "scanName": scan.name,
                "spatialShape": list(shape),
                "volumeCount": volumeCount,
                "qformCode": qformCode,
                "sformCode": sformCode,
                "qformMM": qform.tolist(),
                "voxelSizesMM": sizes.tolist(),
            })

        def sameGrid(leftIndex, rightIndex):
            left, right = geometries[leftIndex], geometries[rightIndex]
            if left["shape"] != right["shape"]: return False
            if not np.allclose(left["sizes"], right["sizes"], rtol=0, atol=spatialToleranceMM): return False
            if not np.allclose(left["origin"], right["origin"], rtol=0, atol=spatialToleranceMM): return False
            if not np.allclose(left["directions"], right["directions"], rtol=0, atol=orientationTolerance): return False

            error = np.linalg.norm(
                ((left["matrix"] - right["matrix"]) @ left["corners"])[:3],
                axis=0,
            ).max()
            return bool(error <= spatialToleranceMM)

        # Comparing every member prevents a chain of small differences from
        # placing mutually incompatible endpoints in the same group.
        groups = []
        for index in indices:
            matches = [
                position for position, members in enumerate(groups)
                if all(sameGrid(index, other) for other in members)
            ]
            if len(matches) > 1:
                raise ValueError(
                    f"Scan {index} matches multiple groups near the tolerance boundary. "
                    "Review the selection/tolerance rather than assigning it arbitrarily."
                )
            if matches: groups[matches[0]].append(index)
            else: groups.append([index])

        def resolveReference(choice, members):
            if type(choice) is int:
                if choice not in members:
                    raise ValueError(
                        f"Reference scan {choice} is not in group {members}. "
                        "Specify one reference per group or explicitly authorize "
                        "cross-geometry alignment with a selected reference index."
                    )
                return choice
            position = {"first": 0, "middle": len(members) // 2, "last": len(members) - 1}[choice]
            return members[position]

        if allowGeometryMismatch:
            reference = resolveReference(referenceRun, indices)
            alignmentGroups = [{
                "groupIndex": 0,
                "scanIndices": list(indices),
                "referenceRule": referenceRun,
                "referenceScanIndex": reference,
                "referenceScanUUID": scans[reference].uuid,
            }]
        else:
            rules = referenceRun if isinstance(referenceRun, list) else [referenceRun] * len(groups)
            if len(rules) != len(groups):
                raise ValueError(f"Expected {len(groups)} reference choices, received {len(rules)}.")

            alignmentGroups = []
            for groupIndex, (members, rule) in enumerate(zip(groups, rules)):
                reference = resolveReference(rule, members)
                alignmentGroups.append({
                    "groupIndex": groupIndex,
                    "scanIndices": list(members),
                    "referenceRule": rule,
                    "referenceScanIndex": reference,
                    "referenceScanUUID": scans[reference].uuid,
                })

        crossGeometry = allowGeometryMismatch and len(groups) > 1
        plan = {
            "schemaVersion": 1,
            "kind": "mriRunAlignmentPreparation",
            "scanIndices": list(indices),
            "requestedReference": deepcopy(referenceRun),
            "allowGeometryMismatch": allowGeometryMismatch,
            "geometryPolicy": {
                "source": "originalQform",
                "spatialToleranceMM": float(spatialToleranceMM),
                "orientationTolerance": float(orientationTolerance),
                "compareSpatialShape": True,
                "compareFieldOfViewCorners": True,
                "groupOrder": "firstAppearanceInSelectedOrder",
                "middleRule": "len(group)//2",
            },
            "scans": scanRecords,
            "geometryGroups": [
                {"groupIndex": index, "scanIndices": list(members)}
                for index, members in enumerate(groups)
            ],
            "alignmentGroups": alignmentGroups,
            "requiresExplicitOutputGrid": crossGeometry,
            "referenceImageResolution": "Required from explicit source/motion product before estimation.",
        }

        if len(groups) > 1:
            message = (
                "Cross-geometry alignment was explicitly authorized. "
                "Application requires an explicit output-grid choice."
                if crossGeometry else
                f"Selected scans contain {len(groups)} acquisition-grid groups. "
                "Alignment will remain independent within each group."
            )
            pglMessages.warning(message, level=1)

        return plan
    
    ################################################################
    # Validate a recorded within-run motion product
    ################################################################
    @staticmethod
    def _validateMotionProduct(record):
        """Validate native or interpreted motion metadata without reading files.

        Returns independent scan records indexed by acquisition UUID.
        Uses the same contract as product inspection and motion plotting.

        Invalid definitions propagate to the owning action runner.
        """
        from .pglMRI import pglMRITransforms

        return pglMRITransforms.validateMotionProduct(record)
    
    ################################################################
    # Resolve explicit motion dependencies and prepare run alignment
    ################################################################
    @staticmethod
    def _prepareRunAlignmentInputs(session, withinRunMotion, scanIndices=None, *, referenceRun="middle", allowGeometryMismatch=False, spatialToleranceMM=0.001, orientationTolerance=0.000001):
        """Prepare run alignment from an explicitly selected motion product.

        Called internally by the run-alignment action. Errors propagate to
        its ordinary action runner, which warns and stops execution.

        withinRunMotion:
            A pglProductSelector for an existing rigidMotion product.
            No latest-product search, implicit producer execution, or fallback
            to another source stage is performed.

        scanIndices:
            Current zero-based indices in session.mri.scans.
            None selects all attached scans, matching the geometry helper.
            The motion product must cover every selected scan.

        Matches product scan records by UUID, not historical scan position.
        Historical indices remain provenance and are returned separately.

        Returns:
            A portable plan identifying geometry groups, reference scans,
            per-scan reference volumes, source requirements, and required
            transform artifacts.

        Source requirements are NOT resolved payloads. A sourceStageID and
        content hash alone do not establish how to open the source image.
        The scientific action must explicitly resolve those stages and verify
        their bytes and geometry before estimating alignment.

        Product resolution automatically records exact input consumption when
        called inside an action. This helper does not append a separate action
        history entry, publish products, execute AFNI, or apply transforms.

        Native AFNI matrix mapping remains unverified. This helper does not
        authorize composition/application of those matrices.
        """
        from .pglPipeline import pglProductSelector
        from .pglSession import pglSession

        if not isinstance(session, pglSession) or session.mri is None:
            raise ValueError("Run alignment requires a session with MRI scans.")
        if not isinstance(withinRunMotion, pglProductSelector):
            raise TypeError("withinRunMotion must be an explicit pglProductSelector.")

        # Resolve through the selector so execution history captures the exact
        # dependency. No file is read by product resolution itself.
        product = withinRunMotion.resolve(session)
        if product is None:
            raise ValueError("The required within-run motion product could not be resolved.")

        productRecord = product.record
        motionByScan = pglActionsMRI._validateMotionProduct(productRecord)

        plan = pglActionsMRI._prepareRunAlignment(
            session.mri,
            scanIndices,
            referenceRun=referenceRun,
            allowGeometryMismatch=allowGeometryMismatch,
            spatialToleranceMM=spatialToleranceMM,
            orientationTolerance=orientationTolerance,
        )

        missing = [
            record["scanIndex"]
            for record in plan["scans"]
            if record["scanUUID"] not in motionByScan
        ]
        if missing:
            raise ValueError(
                f"Within-run motion product {product.name!r} does not cover "
                f"selected MRI scan indices {missing}."
            )

        requirements = []
        requirementsByIndex = {}

        for geometry in plan["scans"]:
            scanIndex = geometry["scanIndex"]
            scanUUID = geometry["scanUUID"]
            motion = motionByScan[scanUUID]
            volumeIndices = motion["volumeIndices"]

            # These are original acquisition indices, not row numbers in a
            # shortened temporal stage or motion-matrix file.
            if any(index >= geometry["volumeCount"] for index in volumeIndices):
                raise ValueError(
                    f"Motion product {product.name!r} refers to an original "
                    f"volume outside scan {scanIndex}'s acquisition."
                )

            referenceVolumeIndex = motion["referenceVolumeIndex"]
            matrixRole = motion["matricesRole"]
            parameterRole = motion["motionParametersRole"]

            requirement = {
                "scanIndex": scanIndex,
                "scanUUID": scanUUID,
                "recordedProducerScanIndex": motion["scanIndex"],
                "sourceStageID": motion["sourceStageID"],
                "sourceIdentity": deepcopy(motion["sourceIdentity"]),
                "volumeIndices": list(volumeIndices),
                "referenceVolumeIndex": referenceVolumeIndex,
                "referenceRow": volumeIndices.index(referenceVolumeIndex),
                "matrixArtifact": deepcopy(productRecord["artifacts"][matrixRole]),
                "motionParametersArtifact": deepcopy(productRecord["artifacts"][parameterRole]),
            }

            requirements.append(requirement)
            requirementsByIndex[scanIndex] = requirement

        alignmentGroups = []

        for group in plan["alignmentGroups"]:
            reference = requirementsByIndex[group["referenceScanIndex"]]

            alignmentGroups.append({
                **deepcopy(group),
                "referenceSourceStageID": reference["sourceStageID"],
                "referenceSourceIdentity": deepcopy(reference["sourceIdentity"]),
                "referenceVolumeIndex": reference["referenceVolumeIndex"],
                "referenceRow": reference["referenceRow"],
            })

        plan["kind"] = "mriRunAlignmentInputs"
        plan["alignmentGroups"] = alignmentGroups
        plan["withinRunMotion"] = {
            "selector": withinRunMotion.toJSONdict(),
            "productID": product.productID,
            "producerActionID": product.producerActionID,
            "matrixConvention": deepcopy(productRecord["metadata"]["matrixConvention"]),
        }
        plan["sourceRequirements"] = requirements
        plan["referenceImageResolution"] = "Explicit source-stage payload resolution required before estimation."
        plan["readyForEstimation"] = False

        return plan
    
    ################################################################
    # Resolve an explicitly declared acquisition or derived image
    ################################################################
    @staticmethod
    def _resolveImageSource(session, source, *, expectedMotionSource=None, cacheDirectory=None):
        """Resolve a verified local image and its original-volume mapping.

        Internal action helper: failures raise to the action runner.
        No fallback source selection, regeneration, or header editing occurs.

        source:
            Acquisition:
                schemaVersion, kind="acquisition", scanUUID

            Derived:
                schemaVersion, kind="derived", scanUUID,
                productID, artifactRole

        Derived image products must declare:
            metadata["imageStages"][artifactRole] = {
                "scanUUID": ...,
                "volumeIndices": [...],
            }

        volumeIndices describes EVERY volume stored in that image, in file
        order. It is not a request to remove or reorder volumes.

        Stage identities:
            acquisition:<scanUUID>
            product:<productID>:<artifactRole>

        expectedMotionSource:
            Optional scan record from a validated motion product. Its scan UUID,
            stage ID, content identity, and original-volume mapping must match.

        cacheDirectory:
            Local cache override for deferred acquisition materialization or
            derived-file retrieval. None uses the source's existing cache
            configuration.

        Returns:
            Runtime dictionary containing localPath, source identity, original
            volume mapping, and the actual on-disk header snapshot.

        Reads file bytes for hashing and headers, not voxel arrays.
        External tools subsequently read voxel data when explicitly invoked.
        Sources must remain unchanged while readers/actions depend on them.

        This establishes source identity and declared volume correspondence.
        Geometry compatibility with acquisition/reference grids must still be
        checked by the scientific action before estimation/application.

        Does not replace scan.localPath, scan._backing, scan._data, or metadata.
        """
        from pathlib import Path

        from .pglExternalTools import _copyProvenance, pglExternalTool
        from .pglMRI import pglMRINiftiMetadata
        from .pglPipeline import pglProductSelector
        from .pglSession import pglSession

        if not isinstance(session, pglSession) or session.mri is None:
            raise ValueError("Image-source resolution requires a session with MRI scans.")

        source = _copyProvenance(source)
        if type(source) is not dict:
            raise TypeError("source must be an explicit image-source dictionary.")
        if type(source.get("schemaVersion")) is not int or source["schemaVersion"] != 1:
            raise ValueError("Unsupported image-source version.")

        kind = source.get("kind")
        expectedFields = {"schemaVersion", "kind", "scanUUID"}
        if kind == "derived":
            expectedFields |= {"productID", "artifactRole"}
        elif kind != "acquisition":
            raise ValueError("Image-source kind must be acquisition or derived.")

        if set(source) != expectedFields:
            raise ValueError("Invalid image-source fields.")

        scanUUID = source["scanUUID"]
        if not isinstance(scanUUID, str) or not scanUUID.strip():
            raise ValueError("Image source requires a scan UUID.")

        matches = [
            (index, scan)
            for index, scan in enumerate(session.mri.scans)
            if scan.uuid == scanUUID
        ]
        if len(matches) != 1:
            raise ValueError("Image source scan UUID is absent or ambiguous in session.mri.scans.")

        scanIndex, scan = matches[0]
        expectedIdentity = None
        volumeIndices = None
        productID = None

        if kind == "acquisition":
            stageID = f"acquisition:{scanUUID}"

            # Access the acquisition backing directly. scan.data could expose
            # an edited working image and is not the source contract here.
            if scan._backing is not None:
                cache = scan._cacheDirectory if cacheDirectory is None else cacheDirectory
                if cache is None or not str(cache).strip():
                    raise ValueError("A local cache directory is required for this acquisition source.")

                path = Path(scan._backing.materialize(cache))
                expectedIdentity = scan._backing.expectedIdentity

            elif scan.localPath:
                path = Path(scan.localPath).expanduser().absolute()
            else:
                raise ValueError(f"Scan {scanIndex} has no acquisition source.")

        else:
            productID = source["productID"]
            role = source["artifactRole"]

            for label, value in (("productID", productID), ("artifactRole", role)):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{label} must be a nonblank string.")

            # Pinned resolution automatically records exact product consumption
            # when invoked inside an action.
            selector = pglProductSelector(productID=productID)
            product = selector.resolve(session)
            if product is None:
                raise ValueError("The explicitly selected image product is unavailable.")

            artifact = product.artifacts.get(role)
            if not isinstance(artifact, dict):
                raise ValueError(f"Image product has no artifact role {role!r}.")

            imageStages = product.metadata.get("imageStages")
            stage = imageStages.get(role) if isinstance(imageStages, dict) else None

            if not isinstance(stage, dict) or set(stage) != {"scanUUID", "volumeIndices"}:
                raise ValueError("Derived image product lacks an explicit image-stage association.")
            if stage["scanUUID"] != scanUUID:
                raise ValueError("Derived image belongs to a different acquisition scan.")

            volumeIndices = stage["volumeIndices"]
            if (
                type(volumeIndices) is not list or not volumeIndices
                or any(type(index) is not int or index < 0 for index in volumeIndices)
                or len(set(volumeIndices)) != len(volumeIndices)
            ):
                raise ValueError("Derived image volumeIndices must contain unique nonnegative original indices.")

            attachmentName = artifact["attachmentName"]
            attachment = session.derivedFiles.get(attachmentName)
            if attachment is None:
                raise ValueError("The declared image payload is not attached; it may have been discarded.")

            expectedIdentity = {key: artifact[key] for key in ("sha256", "size_bytes")}
            if any(attachment.get(key) != value for key, value in expectedIdentity.items()):
                raise ValueError("The image attachment no longer matches its recorded product identity.")

            path = session.getDerivedFile(attachmentName, cacheDirectory=cacheDirectory)
            if path is None:
                raise ValueError("The declared image payload could not be retrieved.")

            path = Path(path)
            stageID = f"product:{productID}:{role}"

        if not path.is_file() or not path.name.lower().endswith((".nii", ".nii.gz")):
            raise ValueError("Resolved image source must be an existing .nii or .nii.gz file.")

        # Verify the actual file, including local-only acquisition sources and
        # callers whose retrieval implementation returned an unverified path.
        identity = pglExternalTool._fileIdentity(path)
        if expectedIdentity is not None and identity != expectedIdentity:
            raise ValueError("Resolved image bytes differ from their expected identity.")

        snapshot = pglMRINiftiMetadata.read(path)
        dimensions = snapshot["fields"].get("dim")
        if (
            not isinstance(dimensions, list) or not dimensions
            or type(dimensions[0]) is not int or dimensions[0] not in (3, 4)
            or len(dimensions) <= dimensions[0]
            or any(type(size) is not int or size <= 0 for size in dimensions[1:dimensions[0] + 1])
        ):
            raise ValueError("Image-source resolution supports valid 3D/4D images only.")

        volumeCount = dimensions[4] if dimensions[0] == 4 else 1

        if kind == "acquisition":
            original = scan.sourceMetadata.get("nifti")
            if original is not None and original != snapshot:
                raise ValueError("Acquisition header differs from its preserved original snapshot.")
            volumeIndices = list(range(volumeCount))

        elif len(volumeIndices) != volumeCount:
            raise ValueError(
                f"Derived image stores {volumeCount} volumes but declares "
                f"{len(volumeIndices)} original-volume indices."
            )

        if expectedMotionSource is not None:
            expected = _copyProvenance(expectedMotionSource)
            if type(expected) is not dict:
                raise TypeError("expectedMotionSource must be a motion scan record.")

            comparisons = {
                "scanUUID": scanUUID,
                "sourceStageID": stageID,
                "sourceIdentity": identity,
                "volumeIndices": volumeIndices,
            }
            for field, actual in comparisons.items():
                if expected.get(field) != actual:
                    raise ValueError(f"Resolved image does not match the motion product's {field}.")

        return {
            "schemaVersion": 1,
            "source": source,
            "scanIndex": scanIndex,
            "scanUUID": scanUUID,
            "sourceStageID": stageID,
            "sourceIdentity": identity,
            "volumeIndices": list(volumeIndices),
            "volumeCount": volumeCount,
            "localPath": path,
            "nifti": snapshot,
            "productID": productID,
        }
        
    ################################################################
    # Apply motion mappings on original acquisition grids
    ################################################################
    @staticmethod
    @operationProgress("Applying MRI motion correction")
    def _applyWithinRunMotion(action, session, productRecord, *, outputName, cacheDirectory=None, runAlignmentRecord=None):
        """Apply exact acquisition-based motion and publish Zarr replacements.

        Supports interpreted within-run motion and optional matching-grid
        between-run alignment derived from that exact motion product.

        Scientific input:
            Original acquisitions with full original volume sequences.
            Existing corrected stages may be replaced, but are not resampled.

        Sampling:
            Output acquisition voxel -> common reference -> run reference
                -> original input volume.

            Cubic SciPy interpolation, constant-zero boundary, float32 output.
            Each output volume is sampled once using the composed mapping.

        Storage/execution:
            Compressed inputs are decompressed once into local working .nii
            files. Volume reads then use NiBabel's memory-mapped proxy.
            Delayed correction tasks are written through the Zarr publisher
            using a synchronous Dask scheduler.

        Progress:
            Reports acquisition/artifact preflight, decompression, correction,
            Zarr computation/writing, and final identity verification.
            Text updates are captured by the action console logger.

        Does not allocate a full corrected run. Individual volumes,
        interpolation workspace, and storage buffers still require memory.

        Registration remains uncommitted until the action succeeds. A failed
        final identity check fails the action; no current stage is switched.

        Successful working .nii files and Zarr bundles remain cached.
        This function does not evict files or close other consumers' readers.
        Acquisition sources must remain immutable throughout processing.
        """
        _progress = currentProgress()
        import os
        import time
        from copy import deepcopy
        from pathlib import Path

        import dask
        import dask.array as da
        import nibabel as nib
        import numpy as np
        from dask import delayed
        from dask.diagnostics import ProgressBar
        from scipy.ndimage import affine_transform

        from .pglData import pglDataMatrix
        from .pglExternalTools import pglExternalTool
        from .pglMRI import pglMRIData, pglMRITransforms
        from .pglPipeline import pglActionStatus
        from .pglSession import pglSession

        if not isinstance(session, pglSession) or session.mri is None:
            raise ValueError("Motion application requires a session with MRI scans.")
        if session.mri._getSession() is not session:
            raise ValueError("MRI collection is not bound to this session.")
        if session._failedActionIDs or session.mri._failedActionIDs:
            raise ValueError("Cannot apply motion to a working object with unresolved failures.")

        if (
            not isinstance(outputName, str) or not outputName.strip()
            or any(character in outputName for character in ("\x00", "\n", "\r"))
        ):
            raise ValueError("outputName must be a nonblank string without NUL or newlines.")

        def localCachePath(value):
            text = os.fspath(value)
            if (
                not isinstance(text, str) or not text.strip()
                or "://" in text or "::" in text
                or any(character in text for character in ("\x00", "\n", "\r"))
            ):
                raise ValueError("cacheDirectory must be an ordinary local path.")
            return Path(text).expanduser().absolute()

        explicitCache = None if cacheDirectory is None else localCachePath(cacheDirectory)

        for entry in session.actionHistory:
            if entry.actionStatus != pglActionStatus.SUCCESS: continue
            if any(
                record.get("name") == outputName
                and record.get("recordStatus") == "committed"
                for record in entry.actionProducts
            ):
                raise ValueError(f"Product {outputName!r} already exists.")

        statusOutput("Motion preflight: validating motion-product metadata.", flush=True)
        motionByScan = pglMRITransforms.validateMotionProduct(productRecord)
        if productRecord["metadata"]["schemaVersion"] != 2:
            raise ValueError("Application requires interpreted version-2 motion.")

        compositePlan = None
        alignmentByScan = {}
        compositeByScan = {}

        if runAlignmentRecord is not None:
            statusOutput("Motion preflight: validating and composing between-run alignment.", flush=True)

            # Preserve exact product, source-stage, reference, and coordinate
            # associations before generating any corrected data.
            compositePlan = pglMRITransforms.composeRunMotion(
                productRecord,
                runAlignmentRecord,
            )
            alignmentByScan = {
                record["scanUUID"]: record
                for record in runAlignmentRecord["metadata"]["scans"]
            }
            compositeByScan = {
                record["scanUUID"]: record
                for record in compositePlan["scans"]
            }
            selectedUUIDs = list(compositeByScan)
        else:
            selectedUUIDs = list(motionByScan)

        # Capture the current stages before computation. They are replacement
        # targets, not implicit scientific inputs for acquisition transforms.
        committedStages, _ = session.mri._getCommittedImageStages()
        replacesByScan = {}

        for scanUUID in selectedUUIDs:
            current = pglMRIData._resolveImageStage(scanUUID, committedStages)
            replacesByScan[scanUUID] = (
                f"acquisition:{scanUUID}" if current is None else current["stageID"]
            )

        positions = {}
        for index, scan in enumerate(session.mri.scans):
            positions.setdefault(scan.uuid, []).append(index)

        for scanUUID in selectedUUIDs:
            if len(positions.get(scanUUID, [])) != 1:
                raise ValueError("A motion acquisition is absent or ambiguous in the current MRI collection.")

        if explicitCache is not None:
            outputCache = explicitCache
        else:
            scanCaches = {
                session.mri.scans[positions[scanUUID][0]]._cacheDirectory
                for scanUUID in selectedUUIDs
                if session.mri.scans[positions[scanUUID][0]]._cacheDirectory
            }
            outputCache = (
                localCachePath(next(iter(scanCaches)))
                if len(scanCaches) == 1 else
                Path.home() / ".cache" / "pgl"
            )

        statusOutput(f"Motion working cache: {outputCache}", flush=True)
        statusOutput("Compressed sources will be expanded once; allow space for uncompressed inputs and corrected Zarr outputs.", flush=True)

        verifiedArtifacts = []
        jobs = []

        def scanLabel(index):
            name = " ".join((session.mri.scans[index].name or "unnamed").split())
            return f"scan {index} {name!r}"

        def readArtifact(ownerRecord, role, expectedShape):
            """Retrieve, verify, and read one exact numerical artifact."""
            statusOutput(f"  Retrieving transform artifact: {role}", flush=True)

            artifact = ownerRecord["artifacts"][role]
            attachment = session.derivedFiles.get(artifact["attachmentName"])
            expected = {key: artifact[key] for key in ("sha256", "size_bytes")}

            if attachment is None or any(attachment.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Motion artifact {role!r} is missing or has a different identity.")

            path = session.getDerivedFile(
                artifact["attachmentName"],
                cacheDirectory=cacheDirectory,
            )
            if path is None:
                raise ValueError(f"Motion artifact {role!r} could not be retrieved.")
            path = Path(path)

            if pglExternalTool._fileIdentity(path) != expected:
                raise ValueError(f"Motion artifact {role!r} failed byte verification.")

            values = np.loadtxt(path, comments="#", ndmin=2)
            if values.shape != expectedShape or not np.isfinite(values).all():
                raise ValueError(f"Motion artifact {role!r} has invalid dimensions or values.")

            verifiedArtifacts.append((path, expected))
            return values

        def requireSourceBytesUnchanged(label):
            """Check actual acquisition and transform bytes around computation."""
            statusOutput(label, flush=True)

            for position, job in enumerate(jobs):
                source = job["source"]
                statusOutput(
                    f"  Verifying acquisition scan {source['scanIndex']} "
                    f"({position} of {len(jobs)} completed).",
                    flush=True,
                )
                if pglExternalTool._fileIdentity(source["localPath"]) != source["sourceIdentity"]:
                    raise ValueError("An acquisition source changed during correction.")

            statusOutput(f"  Verifying {len(verifiedArtifacts)} transform artifacts.", flush=True)
            for path, expected in verifiedArtifacts:
                if pglExternalTool._fileIdentity(path) != expected:
                    raise ValueError("A transform artifact changed during correction.")

        try:
            # Validate all scientific sources/transforms before correction.
            for scanPosition, scanUUID in enumerate(selectedUUIDs):
                motion = motionByScan[scanUUID]
                scanIndex = positions[scanUUID][0]

                statusOutput(
                    f"\nMotion preflight: {scanLabel(scanIndex)} "
                    f"({scanPosition} of {len(selectedUUIDs)} completed).",
                    flush=True,
                )

                if motion["sourceStageID"] != f"acquisition:{scanUUID}":
                    raise ValueError(
                        "This application engine supports original acquisition "
                        "motion sources only."
                    )

                statusOutput("  Resolving original acquisition and checking its identity/header.", flush=True)
                source = progressCall(_progress, 0.04 + 0.16 * scanPosition / len(selectedUUIDs), 0.04 + 0.16 * (scanPosition + 1) / len(selectedUUIDs), f"Preflight scan {scanIndex} — retrieving and verifying acquisition", pglActionsMRI._resolveImageSource, session, {"schemaVersion": 1, "kind": "acquisition", "scanUUID": scanUUID}, expectedMotionSource=motion, cacheDirectory=cacheDirectory)
                count = source["volumeCount"]

                matrices = readArtifact(
                    productRecord, motion["matricesRole"], (count, 12),
                )
                readArtifact(
                    productRecord, motion["motionParametersRole"], (count, 6),
                )

                nativeRows = np.array([
                    np.asarray(transform["matrix"])[:3].reshape(-1)
                    for transform in motion["interpretation"]["nativeTransforms"]
                ])
                if not np.allclose(matrices, nativeRows, rtol=0, atol=1e-10):
                    raise ValueError("Native motion matrices disagree with their interpreted records.")

                statusOutput("  Checking acquisition geometry and sampling transforms.", flush=True)
                image = nib.load(str(source["localPath"]), mmap=True)
                physical = image.get_qform(coded=True)[0]
                sourceFrame = motion["interpretation"]["voxelFrame"]

                if len(image.shape) != 4:
                    raise ValueError("Motion application requires a four-dimensional source.")
                if physical is None or image.header.get_xyzt_units()[0] != "mm":
                    raise ValueError("Application requires a coded millimeter acquisition qform.")
                if (
                    list(image.shape[:3]) != sourceFrame["spatialShape"]
                    or not np.allclose(
                        physical, sourceFrame["voxelToPhysicalMM"],
                        rtol=0, atol=0.0001,
                    )
                ):
                    raise ValueError("Motion interpretation does not describe the acquisition grid.")

                if compositePlan is None:
                    transforms = motion["interpretation"]["voxelTransforms"]
                    alignment = None
                else:
                    alignment = alignmentByScan[scanUUID]

                    if not alignment["isReference"]:
                        rows = readArtifact(
                            runAlignmentRecord, alignment["matricesRole"], (1, 12),
                        )
                        readArtifact(
                            runAlignmentRecord, alignment["motionParametersRole"], (1, 6),
                        )
                        native = np.asarray(
                            alignment["interpretation"]["nativeTransform"]["matrix"]
                        )
                        if not np.allclose(rows.reshape(3, 4), native[:3], rtol=0, atol=1e-10):
                            raise ValueError("Native alignment matrix disagrees with its interpreted record.")

                    transforms = [
                        item["transform"]
                        for item in compositeByScan[scanUUID]["composites"]
                    ]

                if len(transforms) != count:
                    raise ValueError("There must be one sampling transform per source volume.")

                validatedTransforms = []
                for transform in transforms:
                    validated = pglMRITransforms._validateTransform(transform)
                    if (
                        validated["purpose"] != "sampling"
                        or validated["interpretationStatus"] != "validated"
                        or validated["domainFrame"] != sourceFrame
                        or validated["codomainFrame"] != sourceFrame
                    ):
                        raise ValueError("Sampling mappings do not connect the acquisition grid to itself.")
                    validatedTransforms.append(validated)

                statusOutput("  Constructing lazy source metadata; no voxel computation is expected.", flush=True)
                sourceMatrix = pglMRIData.fromNifti(
                    source["localPath"],
                    scanUUID=scanUUID,
                    sourceStageID=source["sourceStageID"],
                    volumeIndices=source["volumeIndices"],
                    name=f"motionSource_scan{scanIndex:02d}",
                )

                # Retain the wrapper immediately so failure cleanup can close it.
                job = {
                    "motion": motion,
                    "source": source,
                    "sourceMatrix": sourceMatrix,
                    "transforms": validatedTransforms,
                    "alignment": alignment,
                    "shape": tuple(image.shape),
                }
                jobs.append(job)

                sourceData = pglMRIData.validate(sourceMatrix)
                if tuple(sourceData.shape) != tuple(image.shape):
                    raise ValueError("Source tensor dimensions differ from the verified image.")
                if not np.allclose(sourceData.attrs["pglMRI"]["affine"], image.affine, rtol=0, atol=1e-8):
                    raise ValueError("Source tensor affine differs from the verified image.")

                statusOutput(f"  Preflight complete: {scanPosition + 1} of {len(selectedUUIDs)} scans.", flush=True)

            # Build uncompressed working sources once. The original source path
            # and compressed content identity remain authoritative provenance.
            for position, job in enumerate(jobs):
                source = job["source"]
                statusOutput(
                    f"\nPreparing volume reads for scan {source['scanIndex']} "
                    f"({position} of {len(jobs)} prepared).",
                    flush=True,
                )
                job["volumePath"] = progressCall(_progress, 0.20 + 0.15 * position / len(jobs), 0.20 + 0.15 * (position + 1) / len(jobs), f"Preparing scan {source['scanIndex']} — uncompressed volume access", pglMRIData.prepareVolumeSource, source["localPath"], cacheDirectory=outputCache)

                preparedImage = nib.load(str(job["volumePath"]), mmap=True)
                if tuple(preparedImage.shape) != job["shape"]:
                    raise ValueError("Prepared volume source has different dimensions.")
                if not np.allclose(
                    preparedImage.affine,
                    job["sourceMatrix"].data.attrs["pglMRI"]["affine"],
                    rtol=0, atol=1e-8,
                ):
                    raise ValueError("Prepared volume source has different geometry.")

                details = job["volumePath"].stat()
                job["volumeStat"] = (details.st_size, details.st_mtime_ns)

            totalVolumes = sum(job["source"]["volumeCount"] for job in jobs)
            statusOutput(
                f"\nReady to correct {totalVolumes} volumes across {len(jobs)} scans.\n"
                "Scientific input: original acquisition. Existing corrected images are replacement targets only.\n"
                f"Zarr output cache: {outputCache / 'mriImages'}",
                flush=True,
            )

            correctionProgress = {
                "completed": 0,
                "lastReport": time.monotonic(),
            }

            def correctVolume(filename, row, mapping, spatialShape, scanIndex, scanCount, scanProgress):
                """Read and sample one uncompressed acquisition volume."""
                if scanProgress["started"] is None:
                    scanProgress["started"] = time.monotonic()
                    statusOutput(f"\nCorrecting scan {scanIndex}: {scanCount} volumes.", flush=True)

                sourceImage = nib.load(filename, mmap=True)
                volume = np.asarray(sourceImage.dataobj[..., row], dtype=np.float32)

                if volume.shape != spatialShape or not np.isfinite(volume).all():
                    raise ValueError("Source volume has invalid dimensions or nonfinite values.")

                corrected = affine_transform(
                    volume,
                    matrix=mapping[:3, :3],
                    offset=mapping[:3, 3],
                    output_shape=spatialShape,
                    output=np.float32,
                    order=3,
                    mode="constant",
                    cval=0,
                    prefilter=True,
                )
                if not np.isfinite(corrected).all():
                    raise ValueError("Motion correction produced nonfinite values.")

                # Synchronous scheduler: these counters are updated serially.
                # Completion order is not assumed to match volume-index order.
                scanProgress["completed"] += 1
                correctionProgress["completed"] += 1
                now = time.monotonic()

                if (
                    scanProgress["completed"] == 1
                    or scanProgress["completed"] == scanCount
                    or now - correctionProgress["lastReport"] >= 10
                ):
                    statusOutput(
                        f"  Scan {scanIndex}: {scanProgress['completed']}/{scanCount} "
                        f"volumes corrected; total {correctionProgress['completed']}/{totalVolumes}; "
                        f"last original volume index {row}; "
                        f"{now - scanProgress['started']:.1f}s elapsed for this scan.",
                        flush=True,
                    )
                    correctionProgress["lastReport"] = now

                # Returning a corrected volume is not the same as completing
                # its Zarr write. The Dask/write messages cover that phase.
                return corrected

            replacements = []

            for job in jobs:
                source = job["source"]
                spatialShape = job["shape"][:3]
                scanProgress = {"completed": 0, "started": None}
                volumes = []

                for row, transform in enumerate(job["transforms"]):
                    task = delayed(correctVolume, pure=False)(
                        str(job["volumePath"]),
                        row,
                        np.asarray(transform["matrix"], dtype=float),
                        spatialShape,
                        source["scanIndex"],
                        source["volumeCount"],
                        scanProgress,
                    )
                    volumes.append(da.from_delayed(task, shape=spatialShape, dtype=np.float32))

                correctedData = job["sourceMatrix"].data.copy(deep=False)
                correctedData.attrs = deepcopy(job["sourceMatrix"].data.attrs)
                correctedData.data = da.stack(volumes, axis=3)
                correctedData.attrs["pglMRI"]["niftiMetadataRole"] = "sourceHeaderSnapshot"

                correctedMatrix = pglDataMatrix(
                    correctedData,
                    name=f"{outputName}_scan{source['scanIndex']:02d}",
                )
                correctedMatrix.provenance = {
                    "operation": "motionCorrection",
                    "scanUUID": source["scanUUID"],
                    "sourceStageID": source["sourceStageID"],
                    "sourceIdentity": deepcopy(source["sourceIdentity"]),
                    "originalNifti": deepcopy(source["nifti"]),
                }
                pglMRIData.validate(correctedMatrix)

                replacements.append({
                    "scanNum": source["scanIndex"],
                    "matrix": correctedMatrix,
                    "inputStageID": source["sourceStageID"],
                    "replacesStageID": replacesByScan[source["scanUUID"]],
                })

            productIDs = [productRecord["productID"]]
            metadata = {
                "schemaVersion": 2,
                "operation": (
                    "withinRunMotionApplication"
                    if runAlignmentRecord is None else
                    "withinAndBetweenRunMotionApplication"
                ),
                "appliedMotionProductID": productRecord["productID"],
                "appliedProducerActionID": productRecord["producerActionID"],
                "appliedTransformIDs": [
                    transform["transformID"]
                    for job in jobs
                    for transform in job["transforms"]
                ],
                "appliedScanUUIDs": list(selectedUUIDs),
                "sources": [
                    {
                        "scanUUID": job["source"]["scanUUID"],
                        "sourceStageID": job["source"]["sourceStageID"],
                        "sourceIdentity": deepcopy(job["source"]["sourceIdentity"]),
                        "volumeIndices": list(job["source"]["volumeIndices"]),
                    }
                    for job in jobs
                ],
                "outputGrid": "originalAcquisition",
                "outputStorage": "pgl.matrix.zarr.v1",
                "spatialInterpolationCountPerOutput": 1,
                "sampling": {
                    "implementation": "scipy.ndimage.affine_transform",
                    "order": 3,
                    "mode": "constant",
                    "cval": 0.0,
                    "prefilter": True,
                    "outputDtype": "float32",
                },
                "execution": {
                    "scheduler": "synchronous",
                    "volumeTaskOrder": "schedulerDefined",
                    "timeChunkSize": 1,
                    "inputAccess": "uncompressedNiftiProxy",
                },
            }

            if runAlignmentRecord is not None:
                productIDs.append(runAlignmentRecord["productID"])
                metadata.update({
                    "appliedRunAlignmentProductID": runAlignmentRecord["productID"],
                    "appliedRunAlignmentProducerActionID": runAlignmentRecord["producerActionID"],
                    "compositePlan": compositePlan,
                })

            progressCall(_progress, 0.35, 0.40, "Verifying source bytes before correction", requireSourceBytesUnchanged, "\nVerifying original source and transform bytes before correction.")

            statusOutput(
                "\nStarting correction and Zarr publication. "
                "The progress bar measures Dask tasks, not bytes or volumes.",
                flush=True,
            )
            with dask.config.set(scheduler="synchronous"):
                with noProgressBar(minimum=0, dt=0.5):
                    registered = progressCall(_progress, 0.40, 0.92, "Correcting volumes and writing Zarr replacements", session.mri.publishImageReplacements, action, replacements, outputName=outputName, cacheDirectory=outputCache, metadata=metadata, inputProductIDs=productIDs)

            statusOutput("\nZarr publication finished. Checking working source files.", flush=True)
            for job in jobs:
                details = job["volumePath"].stat()
                if (details.st_size, details.st_mtime_ns) != job["volumeStat"]:
                    raise ValueError("A prepared volume source changed during correction.")

            progressCall(_progress, 0.92, 0.99, "Final source and transform verification", requireSourceBytesUnchanged, "Performing final original-source and transform byte verification.")
            statusOutput(
                "Motion application validated. Returning to the action runner "
                "for product commitment and retention.",
                flush=True,
            )
            return registered

        finally:
            # These wrappers belong only to this invocation. Published matrices
            # were independently reopened from Zarr and are not closed here.
            for job in jobs:
                job["sourceMatrix"].close()
    
    ################################################################
    # Present a completed motion product
    ################################################################
    @staticmethod
    def _showMotionProduct(session, productID):
        """Display completed motion results without repeating estimation."""
        import matplotlib.pyplot as plt

        product = session.getProduct(productID=productID)
        if product is None: return
        figures = product.display()
        if figures:
            plt.show(block=False)
            
    ################################################################
    # Estimate within-run motion using AFNI
    ################################################################
    class mriEstimateMotionAFNI(pglDeferredAction):
        """Estimate rigid motion and optionally correct original acquisitions.

        scanIndices are zero-based session.mri.scans indices.
        None selects all attached scans; nonfunctional scans are not silently
        skipped or classified from their names.

        This initial action accepts original acquisition sources only.
        No dummy removal or temporal correction is implicit.

        Default execution estimates and applies within-run correction.
        deferApply=True registers motion without creating corrected images.

        Corrected data remain on each original acquisition grid.

        The inherited _run() coordinates _estimate() and _apply().
        """
        scanIndices = List(Int(), default_value=None, allow_none=True, help="Zero-based MRI scan indices; None opens selection during configuration.")
        referenceVolume = Union([Int(min=0), Enum(["mid", "middle", "first", "last"])], default_value="mid", help="Reference within each scan: mid (default), middle, first, last, or a zero-based index.")
        twoPass = Bool(True, help="Use AFNI two-pass motion estimation.")
        outputName = Unicode("motion", help="Name of the motion-estimation product.")
        correctedOutputName = Unicode("motionCorrected", help="Name of the corrected-image product.")
        showMotion = Bool(True, help="Display motion plots after successful execution.")
        afniDirectory = Unicode(default_value=None, allow_none=True, help="AFNI executable directory; None uses ordinary discovery.")
        cacheDirectory = Unicode(default_value=None, allow_none=True, help="Local working-cache override.")
        timeout = Float(default_value=None, allow_none=True, help="Per-scan estimation timeout in seconds.")

        ################################
        # Configure and select scans
        ################################
        def configure(self, session=None, **kwargs) -> bool:
            """Configure motion estimation, prompting when scanIndices is None.

            Explicit or previously configured indices bypass the dialog.
            Pass scanIndices=None with a session to open selection again.

            Reference rules are resolved independently for each acquisition.
            Single-volume acquisitions are silently omitted from the dialog.
            Returns False on cancellation or unavailable selection.
            """
            from .pglMRI import pglMRINiftiMetadata
            from .pglPipeline import pglActionStatus
            from .pglSession import pglSession

            # Cancellation must not leave a previous configured status active.
            self.status = pglActionStatus.INITIALIZED
            self.configureTraits(**kwargs)

            if session is not None and (not isinstance(session, pglSession) or session.mri is None):
                raise ValueError("Scan selection requires a session with MRI scans.")

            if self.scanIndices is not None:
                # Explicit indices support noninteractive pipeline setup.
                indices = self.scanIndices
                if not indices or any(type(index) is not int or index < 0 for index in indices):
                    raise ValueError("scanIndices must contain nonnegative integer indices.")
                if len(set(indices)) != len(indices):
                    raise ValueError("scanIndices must not contain duplicates.")
                if session is not None:
                    pglActionsMRI._selectAcquisitionScans(session.mri, indices)

                super().configure()
                return True

            if session is None:
                pglMessages.warning("Supply session to select MRI scans, or configure explicit scanIndices.", level=1)
                return False

            selectedScans = pglActionsMRI._selectAcquisitionScans(session.mri)
            scanLabels = []
            eligibleIndices = []

            for scanRecord in selectedScans:
                scanIndex = scanRecord["scanIndex"]
                scan = session.mri.scans[scanIndex]

                try:
                    # Read acquisition headers only, never scan.data.
                    snapshot = scan.sourceMetadata.get("nifti")
                    if snapshot is None:
                        if scan._backing is not None:
                            snapshot = pglMRINiftiMetadata.readBacking(scan._backing)
                        elif scan.localPath:
                            snapshot = pglMRINiftiMetadata.read(scan.localPath)
                        else:
                            raise ValueError("No acquisition header or source.")

                    dimensions = snapshot["fields"].get("dim")
                    if (
                        not isinstance(dimensions, list) or not dimensions
                        or type(dimensions[0]) is not int or dimensions[0] not in (3, 4)
                        or len(dimensions) <= dimensions[0]
                        or any(type(size) is not int or size <= 0 for size in dimensions[1:dimensions[0] + 1])
                    ):
                        raise ValueError("Unsupported or invalid image dimensions; expected a valid 3D/4D image.")

                    volumeCount = dimensions[4] if dimensions[0] == 4 else 1

                    # Single-volume scans are normal, but have no within-run
                    # motion to estimate. Do not print a message for each one.
                    if volumeCount < 2: continue

                    referenceIndex = self._resolveReferenceVolume(volumeCount)

                except Exception as error:
                    pglMessages.warning(f"Cannot offer MRI scan {scanIndex}: {error}", level=1)
                    continue

                scanName = scan.name.replace("\n", " ").replace("\r", " ")
                scanLabels.append(f"{scanIndex}: {scanName} — {volumeCount} volumes; reference {referenceIndex} (zero-based)")
                eligibleIndices.append(scanIndex)

            if not scanLabels:
                pglMessages.warning("No scans support within-run motion with the requested reference volume.", level=1)
                return False

            # Import the GUI module only for interactive configuration.
            from .pglDialog import pglDialogs

            selectedPositions = pglDialogs.selectItems(
                scanLabels,
                title=f"Select MRI scans — within-run reference: {self.referenceVolume}",
            )
            if selectedPositions is None:
                pglMessages.message("Motion scan selection cancelled; no processing was started.")
                return False

            # Dialog indices refer to scanLabels, not session.mri.scans.
            indices = [eligibleIndices[position] for position in selectedPositions]
            pglActionsMRI._selectAcquisitionScans(session.mri, indices)
            self.scanIndices = indices

            super().configure()
            return True
        
        ################################################################
        # Estimate through the inherited deferred-action coordinator
        ################################################################
        def _estimate(self, session):
            """Estimate all selected scans, then attach and register outputs."""
            import math
            from pathlib import Path

            import nibabel as nib

            from .pglExternalTools import pglAFNI
            from .pglMRI import pglMRITransforms
            from .pglSession import pglSession

            self._motionRecord = None
            if not isinstance(session, pglSession) or session.mri is None:
                raise ValueError("Motion estimation requires a session with MRI scans.")

            for name in (self.outputName, self.correctedOutputName):
                if not name.strip(): raise ValueError("Product names must not be blank.")
            if self.outputName == self.correctedOutputName:
                raise ValueError("Motion and corrected-image products need different names.")

            existing = {
                record["name"]
                for entry in session.actionHistory
                for record in entry.actionProducts
                if record.get("recordStatus") == "committed"
            }
            requestedNames = {self.outputName}
            if not self.deferApply: requestedNames.add(self.correctedOutputName)
            if requestedNames & existing:
                raise ValueError(f"Output products already exist: {sorted(requestedNames & existing)}")

            # Configuration must resolve interactive selection before execution.
            if self.scanIndices is None:
                raise ValueError("Select MRI scans during configuration or supply explicit scanIndices.")

            # Within-run motion needs acquisition selection, not cross-run
            # geometry grouping or selection of a reference run.
            selectedScans = pglActionsMRI._selectAcquisitionScans(session.mri, self.scanIndices)

            sources = []
            for scanRecord in selectedScans:
                source = pglActionsMRI._resolveImageSource(
                    session,
                    {
                        "schemaVersion": 1,
                        "kind": "acquisition",
                        "scanUUID": scanRecord["scanUUID"],
                    },
                    cacheDirectory=self.cacheDirectory,
                )

                # Preflight every selected scan before launching AFNI.
                # Keep the configured rule unchanged; resolve it per source.
                try:
                    referenceIndex = self._resolveReferenceVolume(source["volumeCount"])
                except ValueError as error:
                    raise ValueError(f"MRI scan {source['scanIndex']}: {error}") from error

                sources.append((source, referenceIndex))

            afni = pglAFNI(directory=self.afniDirectory)
            generated = []
            invocationIDs = []

            for source, referenceIndex in sources:
                scanName = session.mri.scans[source["scanIndex"]].name.replace("\n", " ").replace("\r", " ")
                pglMessages.message(
                    f"Estimating within-run motion: scan {source['scanIndex']} {scanName!r}\n"
                    f"  Aligning {source['volumeCount']} volumes to this scan's original "
                    f"volume {referenceIndex} (zero-based; selection: {self.referenceVolume}).\n"
                    "  No alignment to other scans is performed."
                )

                result = afni.volreg(
                    source["localPath"],
                    referenceVolume=referenceIndex,
                    twoPass=self.twoPass,
                    cacheDirectory=self.cacheDirectory,
                    timeout=self.timeout,
                    sourceStageID=source["sourceStageID"],
                    physicalSpaceID=f"acquisitionPhysical:{source['scanUUID']}",
                )
                if result is None: raise RuntimeError("AFNI motion estimation did not return a result.")

                details = result.provenance["invocation"]["result"]
                if details["sourceIdentity"] != source["sourceIdentity"]:
                    raise ValueError("AFNI input identity differs from the resolved acquisition.")

                image = nib.load(source["localPath"])
                units = image.header.get_xyzt_units()[1]
                secondsPerUnit = {"sec": 1.0, "msec": 0.001, "usec": 0.000001}.get(units)
                tr = None if secondsPerUnit is None else float(image.header.get_zooms()[3] * secondsPerUnit)
                if tr is not None and (not math.isfinite(tr) or tr <= 0): tr = None

                generated.append((source, referenceIndex, result, details, tr))
                invocationIDs.append(result.invocationID)

            files = {}
            records = []

            for source, referenceIndex, result, details, tr in generated:
                roles = {}
                for outputRole in ("matrices", "motionParameters"):
                    output = result.outputs[outputRole]
                    label = "transforms" if outputRole == "matrices" else "motionParameters"
                    role = f"scan{source['scanIndex']:02d}_{label}"

                    # Preserve the source extension; version before the extension.
                    extension = Path(output["localPath"]).suffix
                    baseName = f"{self.outputName}_{role}"
                    attachmentName = f"{baseName}{extension}"
                    suffix = 2
                    while attachmentName in session.derivedFiles:
                        attachmentName = f"{baseName}_v{suffix}{extension}"
                        suffix += 1

                    session.addDerivedFile(
                        attachmentName,
                        Path(output["localPath"]),
                        mediaType="text/plain",
                    )
                    attachment = session.derivedFiles[attachmentName]
                    if any(attachment[key] != output[key] for key in ("sha256", "size_bytes")):
                        raise ValueError("Motion output changed before registration.")

                    files[role] = attachmentName
                    roles[outputRole] = role

                # Products record the actual integer reference used for this
                # acquisition. The action configuration retains the rule.
                records.append({
                    "scanIndex": source["scanIndex"],
                    "scanUUID": source["scanUUID"],
                    "sourceStageID": source["sourceStageID"],
                    "sourceIdentity": source["sourceIdentity"],
                    "volumeIndices": source["volumeIndices"],
                    "referenceVolumeIndex": referenceIndex,
                    "matricesRole": roles["matrices"],
                    "motionParametersRole": roles["motionParameters"],
                    "interpretation": details["interpretation"],
                    "timing": {"trSeconds": tr, "volumeTimesSeconds": None},
                })

            metadata = {
                "schemaVersion": 2,
                "operation": "withinRunMotion",
                "backend": "AFNI",
                "matrixConvention": {
                    "format": "AFNI aff12.1D",
                    "producerOption": "3dvolreg -1Dmatrix_save",
                    "mappingStatus": "interpreted",
                },
                "scans": records,
            }

            # Scientific validation belongs before action success, not in QC.
            artifactRecords = {
                role: {
                    "attachmentName": name,
                    **{key: session.derivedFiles[name][key] for key in ("sha256", "size_bytes")},
                }
                for role, name in files.items()
            }
            pglMRITransforms.validateMotionProduct({
                "kind": "rigidMotion",
                "metadata": metadata,
                "artifacts": artifactRecords,
            })

            self._motionRecord = self.registerProduct(
                self.outputName,
                "rigidMotion",
                session=session,
                files=files,
                metadata=metadata,
                invocationIDs=invocationIDs,
            )
            return session
        
        ################################################################
        # Apply this execution's prepared motion product
        ################################################################
        def _apply(self, session):
            """Apply this execution's exact motion result, not a named fallback."""
            if self._motionRecord is None:
                raise RuntimeError("No motion result was prepared by this execution.")

            pglActionsMRI._applyWithinRunMotion(
                self,
                session,
                self._motionRecord,
                outputName=self.correctedOutputName,
                cacheDirectory=self.cacheDirectory,
            )
            return session

        ################################################################
        # Present results after successful history/product finalization
        ################################################################
        def _afterSuccess(self, session):
            """Show the committed motion product when requested."""
            if self.showMotion and self._motionRecord is not None:
                pglActionsMRI._showMotionProduct(session, self._motionRecord["productID"])
                
        ################################################################
        # Resolve the reference independently for each acquisition
        ################################################################
        def _resolveReferenceVolume(self, volumeCount):
            """Resolve a reference rule to a zero-based source-volume index."""
            if type(volumeCount) is not int or volumeCount < 2:
                raise ValueError("Within-run motion requires at least two volumes.")

            choice = self.referenceVolume
            if type(choice) is int:
                referenceIndex = choice
            elif choice in ("mid", "middle"):
                # For an even count, choose the later of the two middle volumes.
                referenceIndex = volumeCount // 2
            elif choice == "first":
                referenceIndex = 0
            elif choice == "last":
                referenceIndex = volumeCount - 1
            else:
                raise ValueError("referenceVolume must be mid, middle, first, last, or a nonnegative integer.")

            if not 0 <= referenceIndex < volumeCount:
                raise ValueError(f"Reference volume {referenceIndex} is outside this scan's {volumeCount} volumes.")
            return referenceIndex
                
                
    ################################################################
    # Apply explicitly selected transforms
    ################################################################
    class mriApplyTransforms(pglAction):
        """Apply explicit within-run motion, optionally with run alignment.

        transforms:
            One rigidMotion selector, or a list containing exactly one
            rigidMotion selector and one runAlignment selector.

        Composite application verifies that runAlignment was estimated from
        the exact supplied motion product.

        Always samples original acquisition data once on each acquisition grid.
        Does not use previously corrected images or discover pending products.

        Nonlinear warps, mismatched-grid application, and derived source stages
        are not yet supported.
        """

        transforms = Union(
            [Instance(pglProductSelector), List(Instance(pglProductSelector))],
            default_value=None,
            allow_none=True,
            help="Explicit motion selector, or motion and run-alignment selectors.",
        )
        outputName = Unicode("motionCorrected", help="Name of the corrected-image product.")
        cacheDirectory = Unicode(default_value=None, allow_none=True, help="Local working-cache override.")

        def _run(self, session):
            """Resolve explicit products and apply their validated sampling maps."""
            if self.transforms is None:
                raise ValueError("Supply transforms explicitly.")

            selectors = self.transforms if isinstance(self.transforms, list) else [self.transforms]
            if len(selectors) not in (1, 2):
                raise ValueError("Supply one motion selector, or one motion and one run-alignment selector.")

            products = {}
            for selector in selectors:
                product = selector.resolve(session)
                if product is None:
                    raise ValueError("A required transform product could not be resolved.")
                if product.kind not in ("rigidMotion", "runAlignment"):
                    raise ValueError(f"Application is not implemented for {product.kind!r}.")
                if product.kind in products:
                    raise ValueError(f"More than one {product.kind!r} product was supplied.")
                products[product.kind] = product.record

            if "rigidMotion" not in products:
                raise ValueError("Explicitly supply the within-run motion product as well as run alignment.")

            pglActionsMRI._applyWithinRunMotion(
                self,
                session,
                products["rigidMotion"],
                outputName=self.outputName,
                cacheDirectory=self.cacheDirectory,
                runAlignmentRecord=products.get("runAlignment"),
            )
            return session
        
    ################################################################
    # Estimate and register explicit run-reference alignments
    ################################################################
    @staticmethod
    def _estimateRunAlignmentProduct(action, session, withinRunMotion, *, scanIndices=None, referenceRun="middle", outputName="runAlignment", afniDirectory=None, cacheDirectory=None, timeout=None):
        """Estimate one fixed alignment per run within matching-grid groups.

        Uses each run's reference volume recorded in the explicitly selected
        within-run motion product. Does not choose a new reference volume.

        Prints the complete resolved plan before scientific tool execution,
        then reports each estimated pair.

        Initial source support:
            Original acquisitions with the full original volume sequence.
            Derived source stages are explicitly refused, not substituted.

        Reference scan:
            Identity mapping; no registration fit is performed.

        Other scans:
            Estimate run-reference -> common-reference alignment.
            Stored sampling mappings go in the reverse coordinate direction:
            common-reference location -> run-reference sampling location.

        Publishes a runAlignment product only after all estimates succeed.
        Native files remain unchanged and are attached through derivedFiles.
        Failed execution may retain staging/log files for diagnosis.

        Returns the registered product record for immediate application by the
        owning deferred action. The record becomes committed only when that
        action succeeds.

        This helper does not apply transformations or display figures.
        The action's post-success hook will display its committed product.
        """
        import math
        from copy import deepcopy
        from pathlib import Path

        import numpy as np

        from .pglExternalTools import pglAFNI, pglExternalTool
        from .pglMRI import pglMRITransforms

        if not isinstance(outputName, str) or not outputName.strip():
            raise ValueError("outputName must be a nonblank string.")
        if any(character in outputName for character in ("\x00", "\n", "\r")):
            raise ValueError("outputName must not contain NUL or newlines.")

        if timeout is not None:
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("timeout must be positive and finite.")

        # Refuse name collisions before performing expensive work.
        for entry in session.actionHistory:
            if any(
                product.get("name") == outputName
                and product.get("recordStatus") == "committed"
                for product in entry.actionProducts
            ):
                raise ValueError(f"Product {outputName!r} already exists. Choose another outputName.")

        plan = pglActionsMRI._prepareRunAlignmentInputs(
            session,
            withinRunMotion,
            scanIndices,
            referenceRun=referenceRun,
            allowGeometryMismatch=False,
        )

        # Retrieve the same exact recorded product used by preparation, never
        # a new name lookup that could select another product.
        motionProduct = session.getProduct(
            productID=plan["withinRunMotion"]["productID"],
        )
        if motionProduct is None:
            raise ValueError("The prepared within-run motion product is unavailable.")

        motionRecord = motionProduct.record
        if motionRecord["metadata"]["schemaVersion"] != 2:
            raise ValueError("Run alignment requires interpreted version-2 within-run motion.")

        motionByScan = pglMRITransforms.validateMotionProduct(motionRecord)
        requirements = {
            item["scanIndex"]: item
            for item in plan["sourceRequirements"]
        }
        scans = list(session.mri.scans)
        sources = {}

        # Resolve every source before launching a fit. The source must be the
        # exact image used for within-run estimation, not an already corrected
        # image or a guessed image with a similar filename.
        for scanIndex in plan["scanIndices"]:
            requirement = requirements[scanIndex]
            scanUUID = requirement["scanUUID"]
            motion = motionByScan[scanUUID]

            if motion["sourceStageID"] != f"acquisition:{scanUUID}":
                raise ValueError(
                    f"MRI scan {scanIndex} uses derived stage "
                    f"{motion['sourceStageID']!r}. This initial run-alignment "
                    "implementation supports acquisition sources only."
                )

            source = pglActionsMRI._resolveImageSource(
                session,
                {
                    "schemaVersion": 1,
                    "kind": "acquisition",
                    "scanUUID": scanUUID,
                },
                expectedMotionSource=motion,
                cacheDirectory=cacheDirectory,
            )

            # Match the actual source geometry to the motion interpretation.
            # Matching source identity is necessary but does not validate
            # arbitrary interpreted metadata stored alongside it.
            frame = motion["interpretation"]["voxelFrame"]
            snapshot = source["nifti"]
            fields = snapshot["fields"]
            if fields["xyzt_units"] & 7 != 2:
                raise ValueError("This AFNI run-alignment adapter requires millimeter spatial units.")

            actualQform = np.asarray(snapshot["transforms"]["qform"], dtype=float)
            if (
                list(fields["dim"][1:4]) != frame["spatialShape"]
                or frame["geometryID"] != motion["sourceStageID"]
                or not np.allclose(
                    actualQform, frame["voxelToPhysicalMM"],
                    rtol=0, atol=0.0001,
                )
            ):
                raise ValueError(f"MRI scan {scanIndex}: interpreted motion geometry differs from its source.")

            sources[scanIndex] = source

        def scanLabel(index):
            # Keep each report on one line even if source labels contain newlines.
            name = scans[index].name.replace("\n", " ").replace("\r", " ")
            return f"scan {index} {name!r}"

        def volumeLabel(index):
            requirement = requirements[index]
            original = requirement["referenceVolumeIndex"]
            row = requirement["referenceRow"]
            text = f"original volume {original}"
            if row != original:
                text += f" (source-file volume {row})"
            return text

        pglMessages.message(
            f"Estimating cross-run alignment from motion product {motionProduct.name!r}.\n"
            "Scan and volume indices below are zero-based.\n"
            "Each run uses its recorded within-run reference volume."
        )

        for group in plan["alignmentGroups"]:
            referenceIndex = group["referenceScanIndex"]
            pglMessages.print(f"\nCross-run alignment — group {group['groupIndex']}")
            pglMessages.print(f"  Selected MRI scans: {group['scanIndices']}")
            pglMessages.print(
                f"  Common reference: {scanLabel(referenceIndex)}, "
                f"{volumeLabel(referenceIndex)}"
            )

            for index in group["scanIndices"]:
                if index == referenceIndex:
                    pglMessages.print(
                        f"  {scanLabel(index)}: common reference; "
                        "using identity, no registration fit"
                    )
                else:
                    pglMessages.print(
                        f"  {scanLabel(index)}, {volumeLabel(index)} "
                        f"→ {scanLabel(referenceIndex)}, {volumeLabel(referenceIndex)}"
                    )

        pglMessages.print(
            "\nNo corrected images are generated during estimation.\n"
            "Saved sampling maps go from the common reference to each run's reference."
        )

        afni = pglAFNI(directory=afniDirectory)
        generated = []
        producingInvocations = []
        executionID = action.actionHistory[-1].actionID

        for group in plan["alignmentGroups"]:
            referenceIndex = group["referenceScanIndex"]
            referenceSource = sources[referenceIndex]
            referenceRequirement = requirements[referenceIndex]

            # This identifies the registration group's coordinate reference.
            # It is established explicitly for these matching acquisition grids,
            # not inferred globally from all scans using RAS coordinates.
            physicalSpaceID = (
                f"runAlignment:{executionID}:group{group['groupIndex']}"
            )

            for index in group["scanIndices"]:
                source = sources[index]
                requirement = requirements[index]
                output = None

                if index == referenceIndex:
                    referenceMotion = motionByScan[source["scanUUID"]]
                    oldFrame = referenceMotion["interpretation"]["voxelFrame"]

                    frame = pglMRITransforms.makeFrame(
                        geometryID=source["sourceStageID"],
                        physicalSpaceID=physicalSpaceID,
                        spatialShape=oldFrame["spatialShape"],
                        voxelToPhysicalMM=oldFrame["voxelToPhysicalMM"],
                        representation="voxelIndices",
                    )
                    voxelTransform = pglMRITransforms.makeTransform(
                        np.eye(4),
                        domainFrame=frame,
                        codomainFrame=frame,
                        purpose="sampling",
                        interpretationStatus="validated",
                        adapter="pgl.runAlignment.referenceIdentity.v1",
                    )
                    alignment = None
                    pglMessages.message(
                        f"Group {group['groupIndex']}: {scanLabel(index)} "
                        "is the reference; no fit needed."
                    )

                else:
                    pglMessages.message(
                        f"Group {group['groupIndex']}: aligning "
                        f"{scanLabel(index)}, {volumeLabel(index)} "
                        f"to {scanLabel(referenceIndex)}, {volumeLabel(referenceIndex)}."
                    )

                    output = afni.alignRunReference(
                        source["localPath"],
                        referenceSource["localPath"],
                        inputVolume=requirement["referenceRow"],
                        referenceVolume=referenceRequirement["referenceRow"],
                        inputStageID=source["sourceStageID"],
                        referenceStageID=referenceSource["sourceStageID"],
                        physicalSpaceID=physicalSpaceID,
                        cacheDirectory=cacheDirectory,
                        timeout=timeout,
                    )
                    if output is None:
                        raise RuntimeError(f"Run alignment failed for MRI scan {index}.")

                    details = output.provenance["invocation"]["result"]

                    for inputName, expectedSource in (
                        ("moving", source),
                        ("reference", referenceSource),
                    ):
                        actual = details["inputs"][inputName]
                        if (
                            actual["sourceStageID"] != expectedSource["sourceStageID"]
                            or any(
                                actual[key] != expectedSource["sourceIdentity"][key]
                                for key in ("sha256", "size_bytes")
                            )
                        ):
                            raise ValueError("Alignment tool inputs differ from the prepared source identities.")

                    alignment = details["alignment"]
                    voxelTransform = alignment["voxelTransform"]
                    producingInvocations.append(output.invocationID)

                generated.append({
                    "groupIndex": group["groupIndex"],
                    "scanIndex": index,
                    "scanUUID": source["scanUUID"],
                    "sourceStageID": source["sourceStageID"],
                    "sourceIdentity": deepcopy(source["sourceIdentity"]),
                    "volumeIndices": list(source["volumeIndices"]),
                    "referenceVolumeIndex": requirement["referenceVolumeIndex"],
                    "referenceRow": requirement["referenceRow"],
                    "commonReferenceScanIndex": referenceIndex,
                    "commonReferenceScanUUID": referenceSource["scanUUID"],
                    "commonReferenceSourceStageID": referenceSource["sourceStageID"],
                    "commonReferenceVolumeIndex": referenceRequirement["referenceVolumeIndex"],
                    "isReference": index == referenceIndex,
                    "voxelTransform": voxelTransform,
                    "interpretation": alignment,
                    "_toolResult": output,
                })

            pglMessages.message(f"Finished estimates for geometry group {group['groupIndex']}.")

        # No display-time discovery of invalid transformations: check all
        # group summaries before any product is registered.
        summaries = []
        for group in plan["alignmentGroups"]:
            entries = [
                item for item in generated
                if item["groupIndex"] == group["groupIndex"]
            ]
            summaries.append(pglMRITransforms.runAlignmentSummary(
                [item["voxelTransform"] for item in entries],
                scanIndices=[item["scanIndex"] for item in entries],
                referenceScanIndex=group["referenceScanIndex"],
                scanNames=[scans[item["scanIndex"]].name for item in entries],
            ))

        # Verify all sources again, including reference-only single-run groups.
        for source in sources.values():
            if pglExternalTool._fileIdentity(source["localPath"]) != source["sourceIdentity"]:
                raise ValueError("An acquisition source changed during cross-run estimation.")

        files = {}
        scanRecords = []

        for item in generated:
            output = item["_toolResult"]
            record = {key: deepcopy(value) for key, value in item.items() if key != "_toolResult"}
            record["matricesRole"] = None
            record["motionParametersRole"] = None

            if output is not None:
                for toolRole, recordField in (
                    ("matrices", "matricesRole"),
                    ("motionParameters", "motionParametersRole"),
                ):
                    artifact = output.outputs[toolRole]
                    label = "transforms" if toolRole == "matrices" else "motionParameters"
                    role = f"scan{item['scanIndex']:02d}_{label}"

                    extension = Path(artifact["localPath"]).suffix
                    baseName = f"{outputName}_{role}"
                    attachmentName = f"{baseName}{extension}"
                    suffix = 2
                    while attachmentName in session.derivedFiles:
                        attachmentName = f"{baseName}_v{suffix}{extension}"
                        suffix += 1

                    session.addDerivedFile(
                        attachmentName,
                        Path(artifact["localPath"]),
                        mediaType="text/plain",
                    )
                    attachment = session.derivedFiles[attachmentName]
                    if any(
                        attachment[key] != artifact[key]
                        for key in ("sha256", "size_bytes")
                    ):
                        raise ValueError("Run-alignment output changed before registration.")

                    files[role] = attachmentName
                    record[recordField] = role

            # Identity mappings are explicit records, not fabricated AFNI files.
            scanRecords.append(record)

        product = action.registerProduct(
            outputName,
            "runAlignment",
            session=session,
            files=files,
            inputProductIDs=[motionProduct.productID],
            metadata={
                "schemaVersion": 1,
                "operation": "runReferenceAlignment",
                "backend": "AFNI",
                "withinRunMotionProductID": motionProduct.productID,
                "withinRunMotionProducerActionID": motionProduct.producerActionID,
                "plan": plan,
                "scans": scanRecords,
                "summaries": summaries,
                "application": "notPerformedByEstimation",
                "description": (
                    "One fixed alignment per run, estimated using the reference "
                    "volume recorded by that run's within-run motion estimate. "
                    "Each stored sampling mapping starts in the common group "
                    "reference and ends in the corresponding run reference. "
                    "No corrected image was generated by estimation."
                ),
            },
            invocationIDs=producingInvocations,
        )

        pglMessages.message(
            f"Registered run-alignment product {outputName!r}: "
            f"{len(scanRecords)} scans in {len(plan['alignmentGroups'])} groups.\n"
            "Estimation is complete; image application has not been performed by this helper."
        )
        return product
    
    ################################################################
    # Estimate fixed run-to-reference alignment using AFNI
    ################################################################
    class mriEstimateRunAlignmentAFNI(pglDeferredAction):
        """Align each run's recorded motion reference to a group reference.

        Explicitly consumes a within-run motion product.
        Uses each scan's recorded reference volume—not an assumed volume zero.

        Matching acquisition grids form independent groups.
        referenceRun selects the common scan independently within each group:
            first, middle, last, explicit session.mri.scans index,
            or a list with one selection per group.

        Default execution estimates and applies within-run plus run alignment
        in one interpolation from the original acquisitions.
        deferApply=True registers alignment without generating corrected images.

        Initial support is original acquisition sources and matching-grid groups.
        Cross-geometry alignment/application is not implemented by this action.

        Automatically displays alignment summaries after successful execution
        unless showAlignment=False.
        """

        withinRunMotion = Instance(
            pglProductSelector,
            default_value=None,
            allow_none=True,
            help="Required explicit within-run motion product selector.",
        )
        scanIndices = List(
            Int(),
            default_value=None,
            allow_none=True,
            help="Zero-based session.mri.scans indices; None selects all.",
        )
        referenceRun = Union(
            [
                Int(),
                Unicode(),
                List(Union([Int(), Unicode()])),
            ],
            default_value="middle",
            help="first/middle/last, explicit MRI scan index, or one choice per geometry group.",
        )
        outputName = Unicode("runAlignment", help="Name of the estimated run-alignment product.")
        correctedOutputName = Unicode("motionAndRunAligned", help="Name of the composite-corrected image product.")
        showAlignment = Bool(True, help="Display between-run alignment summaries after success.")
        afniDirectory = Unicode(default_value=None, allow_none=True, help="AFNI executable directory.")
        cacheDirectory = Unicode(default_value=None, allow_none=True, help="Local working-cache override.")
        timeout = Float(default_value=None, allow_none=True, help="Timeout in seconds for each alignment fit.")

        def _estimate(self, session):
            """Estimate from explicit recorded references and retain exact inputs."""
            from .pglSession import pglSession

            self._runAlignmentRecord = None
            self._inputMotionRecord = None

            if not isinstance(session, pglSession) or session.mri is None:
                raise ValueError("Run alignment requires a session with MRI scans.")
            if self.withinRunMotion is None:
                raise ValueError(
                    "Supply withinRunMotion explicitly, for example "
                    "pglProductSelector('motion', kind='rigidMotion')."
                )
            if self.outputName == self.correctedOutputName:
                raise ValueError("Alignment and corrected-image products need different names.")

            for name in (self.outputName, self.correctedOutputName):
                if (
                    not isinstance(name, str) or not name.strip()
                    or any(character in name for character in ("\x00", "\n", "\r"))
                ):
                    raise ValueError("Product names must be nonblank and contain no NUL or newlines.")

            # Refuse corrected-output collisions before estimating anything.
            if not self.deferApply:
                for entry in session.actionHistory:
                    if any(
                        record.get("name") == self.correctedOutputName
                        and record.get("recordStatus") == "committed"
                        for record in entry.actionProducts
                    ):
                        raise ValueError(f"Product {self.correctedOutputName!r} already exists.")

            motion = self.withinRunMotion.resolve(session)
            if motion is None or motion.kind != "rigidMotion":
                raise ValueError("withinRunMotion must resolve to a rigidMotion product.")

            self._inputMotionRecord = motion.record
            self._runAlignmentRecord = pglActionsMRI._estimateRunAlignmentProduct(
                self,
                session,
                self.withinRunMotion,
                scanIndices=self.scanIndices,
                referenceRun=self.referenceRun,
                outputName=self.outputName,
                afniDirectory=self.afniDirectory,
                cacheDirectory=self.cacheDirectory,
                timeout=self.timeout,
            )

            from .pglMRI import pglMRITransforms

            # Validate the complete recorded product inside scientific
            # execution, not for the first time in the plotting hook.
            pglMRITransforms.validateRunAlignmentProduct(self._runAlignmentRecord)

            if (
                self._runAlignmentRecord["metadata"]["withinRunMotionProductID"]
                != self._inputMotionRecord["productID"]
            ):
                raise ValueError("The resolved motion dependency changed during estimation.")

            return session

        def _apply(self, session):
            """Apply this execution's alignment with its exact motion dependency."""
            if self._runAlignmentRecord is None or self._inputMotionRecord is None:
                raise RuntimeError("Run-alignment inputs were not prepared by this execution.")

            pglActionsMRI._applyWithinRunMotion(
                self,
                session,
                self._inputMotionRecord,
                outputName=self.correctedOutputName,
                cacheDirectory=self.cacheDirectory,
                runAlignmentRecord=self._runAlignmentRecord,
            )
            return session

        def _afterSuccess(self, session):
            """Display the committed alignment product, optionally after application."""
            if not self.showAlignment or self._runAlignmentRecord is None:
                return

            import matplotlib.pyplot as plt

            product = session.getProduct(
                productID=self._runAlignmentRecord["productID"],
            )
            if product is None: return

            figures = product.display(cacheDirectory=self.cacheDirectory)
            if figures: plt.show(block=False)

    ############################            
    # GLMsingle
    ############################            
    class GLMsingle(pglAction):
        """Fit GLMsingle using matched runs and current MRI data.

        All scientific preparation, fitting, conversion, and publication
        belongs to this action.

        Betas and diagnostics are analysis products, not image replacements.

        Currently supports explicit Type B selection. Type C/D execution and
        output handling remain to be validated before they are supported.
        """

        scanIndices = List(Int(min=0),default_value=None,allow_none=True,help="Ordered matched MRI scan indices; None preserves saved correspondence order.")
        taskName = Unicode(default_value=None,allow_none=True,help="Behavioral task selection; None uses existing main-task selection.")
        segmentNum = Int(default_value=0,min=0,allow_none=True,help="Zero-based onset segment; None selects trial START.")
        stimDur = Float(2.0,help="Explicit stimulus duration in seconds.")
        model = Enum(["B"],default_value=None,allow_none=True,help="Explicit model selection. Currently only B is supported; no automatic selection.")
        glmOptions = Dict(default_value={},help="Additional GLMsingle options; model flags and output routing are action-managed.")
        retainIntermediateModels = Bool(False,help="Also retain Type A matrices and the full candidate-HRF score arrays.")
        outputName = Unicode("GLMsingle",help="Name of the GLMsingle analysis product.")
        cacheDirectory = Unicode(default_value=None,allow_none=True,help="Local working-cache override; None uses ~/.cache/pgl.")
        showFigures = Bool(True, help="Display GLMsingle figures after successful execution.")
        
        @operationProgress("GLMsingle analysis")
        def _run(self, session):
            """Prepare, fit, validate, and publish one GLMsingle analysis.

            Uses the existing action execution/console-capture infrastructure.
            Does not create another progress display or publish image replacements.

            Current implementation:
                Explicit Type B selection.
                Current committed MRI replacements, otherwise acquisition data.
                Independent writable numerical fit inputs.
                Validated chronological single-trial output mapping.
                Lazy Zarr analysis outputs and explicit support artifacts.

            Scientific publication occurs here, not in _afterSuccess().
            The action runner determines final product commitment.
            """
            import inspect
            from copy import deepcopy

            from .pglExternalTools import getExecutionRecorder
            from .pglSession import pglSession

            progress = currentProgress()
            prepared = None
            fitted = None

            if not isinstance(session, pglSession) or session.mri is None:
                raise ValueError("GLMsingle requires a session with MRI data.")
            if session.mri._getSession() is not session:
                raise ValueError("MRI collection is not bound to this session.")
            if session._failedActionIDs or session.mri._failedActionIDs:
                raise ValueError("Cannot fit a working object with unresolved action failures.")

            recorder = getExecutionRecorder()
            if (
                not getattr(self, "_executionInProgress", False)
                or recorder is None
                or not self.actionHistory
                or recorder.actionID != self.actionHistory[-1].actionID
            ):
                raise RuntimeError("Run GLMsingle through the ordinary action runner, not by calling _run() directly.")

            if self.model != "B":
                raise ValueError(
                    "Explicitly select model='B' to use the currently validated Type B path. "
                    "Type C/D support is not implemented yet; no model is selected automatically."
                )
            if (
                not self.outputName.strip()
                or any(character in self.outputName for character in ("\x00", "\n", "\r"))
            ):
                raise ValueError("outputName must be nonblank without NUL or newlines.")

            if any(
                record.get("name") == self.outputName
                and record.get("recordStatus") == "committed"
                for entry in session.actionHistory
                for record in entry.actionProducts
            ):
                raise ValueError(f"Product {self.outputName!r} already exists.")
            if any(record["name"] == self.outputName for record in recorder.exportProducts()):
                raise ValueError("This execution already registered the requested outputName.")

            # Scientific model flags and output routing have dedicated controls.
            # Reject conflicts instead of overriding the user's supplied options.
            reservedOptions = {
                "wantlibrary",
                "wantglmdenoise",
                "wantfracridge",
                "wantlss",
                "wantfileoutputs",
                "wantmemoryoutputs",
                "wanthdf5",
            }
            conflicts = reservedOptions.intersection(self.glmOptions)
            if conflicts:
                raise ValueError(
                    f"These glmOptions are controlled by the supported model/output path: "
                    f"{sorted(conflicts)}."
                )
            if any(not isinstance(name, str) for name in self.glmOptions):
                raise TypeError("GLMsingle option names must be strings.")

            params = {
                **deepcopy(self.glmOptions),
                "wantlibrary": 1,
                "wantglmdenoise": 0,
                "wantfracridge": 0,
                "wantlss": 0,
            }

            # Fail on installation/API problems before preparing or loading MRI.
            # Installation itself remains the external tool adapter's concern.
            from glmsingle.glmsingle import GLM_single

            inspect.signature(GLM_single).bind(params={})
            inspect.signature(GLM_single.fit).bind(
                None,
                [],
                [],
                stimdur=self.stimDur,
                tr=1.0,
                outputdir=None,
                figuredir=None,
            )

            pglMessages.message(
                f"GLMsingle analysis: {self.outputName!r}\n"
                "Explicit model: B — HRF fitting, without GLMdenoise or fractional ridge.\n"
                "Inputs: current committed MRI images, otherwise acquisitions.\n"
                "Outputs: analysis matrices; current MRI images will not be replaced."
            )

            try:
                prepared = progressCall(
                    progress, 0.0, 0.1,
                    "Preparing GLMsingle designs and input provenance",
                    self._prepareInputs,
                    session,
                    scanIndices=self.scanIndices,
                    taskName=self.taskName,
                    segmentNum=self.segmentNum,
                    stimDur=self.stimDur,
                )

                fitted = progressCall(
                    progress, 0.1, 0.8,
                    "Loading and fitting GLMsingle",
                    self._fitPreparedInputs,
                    prepared,
                    params=params,
                    cacheDirectory=self.cacheDirectory,
                )

                # Default: final single-trial betas and compact fit diagnostics.
                # Full HRF-candidate scores and Type A matrices are optional.
                retainRoles = [
                    "typeb.betasmd",
                    "typeb.R2",
                    "typeb.R2run",
                    "typeb.meanvol",
                    "typeb.HRFindex",
                    "typeb.HRFindexrun",
                ]
                if self.retainIntermediateModels:
                    retainRoles = list(fitted["matrices"])

                progressCall(
                    progress, 0.8, 1.0,
                    "Storing and registering GLMsingle analysis",
                    self._publishResults,
                    session,
                    prepared,
                    fitted,
                    outputName=self.outputName,
                    retainRoles=retainRoles,
                )

                return session

            finally:
                # Published outputs were independently reopened from Zarr.
                # Close only the temporary conversion wrappers and input views.
                if fitted is not None:
                    for matrix in fitted["matrices"].values():
                        matrix.close()
                if prepared is not None:
                    for matrix in prepared["matrices"]:
                        matrix.close()
        def _prepareInputs(self, session, *, scanIndices=None, taskName=None, segmentNum=0, stimDur=2.0):
            """Prepare shared-condition designs and lazy MRI input matrices.

            Uses existing scan/run matches and established volume correspondence.
            Does not infer alignment, fit models, compute MRI arrays, or write files.

            scanIndices:
                Optional ordered list/tuple of matched MRI scan indices.
                None preserves saved correspondence order.

            segmentNum:
                Zero-based onset segment. None selects trial START.

            stimDur:
                Explicit positive duration in seconds; need not be a multiple of TR.

            Conditions:
                Exact recorded source-image identity defines shared columns:
                filesystemPrefix, filename, datasetPath, imageAxis, imageIndex.
                Requires complete source-image provenance; no name/ID fallback.
                Assumes the referenced source dataset was unchanged between runs.
                Conditions absent from a run retain zero columns.

            Beta ordering:
                Selected run order, then onset-volume order within each run.
                Every presentation receives an explicit betaIndex.

            Missing presentations, unmapped volumes, duplicate trial indices,
            inconsistent identities, and onset collisions raise rather than
            silently changing the design.

            Returns a runtime dictionary containing:
                design, matrices, conditions, trials, runs,
                stimDur, tr, nVoxels, nTrials, conditionIdentityMethod.

            Returned matrices are lazy analysis views. The caller must keep
            their source session available and close the views when finished.
            """
            from copy import deepcopy
            from numbers import Integral, Real

            import numpy as np

            if session.mri is None:
                raise ValueError("GLMsingle requires an MRI collection.")

            def validIndex(value):
                return not isinstance(value, (bool, np.bool_)) and isinstance(value, Integral) and value >= 0

            def finiteNumber(value):
                return not isinstance(value, (bool, np.bool_)) and isinstance(value, Real) and np.isfinite(value)

            if not finiteNumber(stimDur) or stimDur <= 0:
                raise ValueError("stimDur must be positive and finite.")

            matches = list(session.getScanRunMatches())
            if not matches:
                raise ValueError("No scan/run matches are configured.")

            byScan = {}
            usedRuns = set()

            for scanIndex, runIndex in matches:
                if (
                    not validIndex(scanIndex) or scanIndex >= len(session.mri.scans)
                    or not validIndex(runIndex) or runIndex >= len(session.runs)
                ):
                    raise ValueError("Invalid scan/run correspondence indices.")

                scanIndex, runIndex = int(scanIndex), int(runIndex)
                if scanIndex in byScan or runIndex in usedRuns:
                    raise ValueError("Scan/run correspondence contains repeated endpoints.")

                byScan[scanIndex] = runIndex
                usedRuns.add(runIndex)

            if scanIndices is None:
                selectedScans = list(byScan)
            else:
                if not isinstance(scanIndices, (list, tuple)) or not scanIndices:
                    raise ValueError("scanIndices must be a nonempty list/tuple or None.")
                if any(not validIndex(index) for index in scanIndices):
                    raise ValueError("scanIndices must contain nonnegative integer indices.")

                selectedScans = [int(index) for index in scanIndices]
                if len(set(selectedScans)) != len(selectedScans):
                    raise ValueError("scanIndices must not contain duplicates.")
                if any(index not in byScan for index in selectedScans):
                    raise ValueError("Every selected scan must have a saved behavioral match.")

            matrices, runRecords, conditions, allTrials = [], [], [], []
            conditionColumns, stimulusSources = {}, {}
            trialsByRun = []
            commonTR, commonShape, nVoxels = None, None, None
            identityFields = ("filesystemPrefix", "filename", "datasetPath", "imageAxis", "imageIndex")

            try:
                for fitRunIndex, scanIndex in enumerate(selectedScans):
                    runIndex = byScan[scanIndex]
                    scan = session.mri.scans[scanIndex]
                    run = session.runs[runIndex]

                    info = run.getTrialsInfo(
                        taskName=taskName,
                        segmentNum=segmentNum,
                        includeBlank=False,
                    )
                    if info is None:
                        raise ValueError(f"Run {runIndex}: trial information could not be selected.")
                    if info.groupBy != "stimulus":
                        raise ValueError("GLMsingle preparation currently requires stimulus grouping.")

                    # Blank exclusion is intentional. Other omissions must not
                    # silently turn missing presentations into baseline periods.
                    for reason, indices in info.omittedTrialNums.items():
                        if reason != "blankStimulus" and len(indices):
                            raise ValueError(f"Run {runIndex}: omitted presentations ({reason}): {indices}")

                    groupedFieldNames = (
                        "stimulusIds",
                        "groupNames",
                        "trialNums",
                        "volumes",
                        "onsetTimestamps",
                        "startTimes",
                        "sourceInfo",
                        "imageIndices",
                    )
                    for fieldName in groupedFieldNames:
                        values = getattr(info, fieldName, None)
                        if values is None or not hasattr(values, "__len__") or len(values) != info.nGroups:
                            raise ValueError(f"Run {runIndex}: inconsistent grouped field {fieldName}.")

                    matrix = session.mri.getDataMatrix(scanNum=scanIndex)
                    if matrix is None:
                        raise ValueError(f"Scan {scanIndex}: current MRI data is unavailable.")
                    matrices.append(matrix)

                    data = matrix.requireData()
                    if tuple(data.dims) != ("voxel", "volume"):
                        raise ValueError("MRI data must have dimensions (voxel, volume).")
                    if data.shape[0] <= 0 or data.shape[1] <= 0:
                        raise ValueError("MRI data must have nonempty voxel and volume axes.")

                    mriMetadata = data.attrs["pglMRI"]
                    if mriMetadata["scanUUID"] != scan.uuid:
                        raise ValueError("Selected matrix identifies another acquisition.")

                    tr = mriMetadata["trSeconds"]
                    if not finiteNumber(tr) or tr <= 0:
                        raise ValueError(f"Scan {scanIndex}: a positive header-derived TR is required.")

                    shape = list(mriMetadata["spatialShape"])
                    if (
                        len(shape) != 3
                        or any(not validIndex(size) or size == 0 for size in shape)
                        or int(np.prod(shape)) != data.shape[0]
                    ):
                        raise ValueError("MRI spatial shape disagrees with the voxel dimension.")

                    # Structural checks only; anatomical correspondence is an
                    # upstream processing responsibility.
                    if commonTR is None:
                        commonTR = float(tr)
                        commonShape = shape
                        nVoxels = int(data.shape[0])
                    elif (
                        not np.isclose(tr, commonTR, rtol=0, atol=1e-6)
                        or shape != commonShape
                        or data.shape[0] != nVoxels
                    ):
                        raise ValueError("Selected runs require matching TR, voxel count, and spatial shape.")

                    volumeIndices = list(mriMetadata["volumeIndices"])
                    if (
                        len(volumeIndices) != data.shape[1]
                        or any(not validIndex(index) for index in volumeIndices)
                        or len(set(volumeIndices)) != len(volumeIndices)
                    ):
                        raise ValueError("Selected MRI data has invalid original-volume indices.")

                    volumeIndices = [int(index) for index in volumeIndices]
                    if any(right != left + 1 for left, right in zip(volumeIndices, volumeIndices[1:])):
                        raise ValueError("GLMsingle requires consecutive, ordered MRI volumes.")

                    # Established behavioral correspondence supplies original
                    # acquisition indices. Map them to the selected data rows.
                    volumeRows = {original: row for row, original in enumerate(volumeIndices)}
                    rows = []
                    seenTrials, seenStimulusIDs = set(), set()

                    for groupIndex in range(info.nGroups):
                        trialNums = info.trialNums[groupIndex]
                        volumes = info.volumes[groupIndex]
                        timestamps = info.onsetTimestamps[groupIndex]
                        startTimes = info.startTimes[groupIndex]

                        if any(len(values) != len(trialNums) for values in (volumes, timestamps, startTimes)):
                            raise ValueError("Grouped onset/trial fields have inconsistent lengths.")
                        if len(trialNums) == 0: continue

                        label = f"Run {runIndex}, group {groupIndex}"
                        stimulusID = info.stimulusIds[groupIndex]
                        if not isinstance(stimulusID, str) or not stimulusID.strip():
                            raise ValueError(f"{label}: every presented stimulus needs a stimulusId.")
                        if stimulusID in seenStimulusIDs:
                            raise ValueError(f"Run {runIndex}: multiple groups share stimulusId {stimulusID!r}.")
                        seenStimulusIDs.add(stimulusID)

                        sourceInfo = info.sourceInfo[groupIndex]
                        if not isinstance(sourceInfo, dict):
                            raise ValueError(f"{label}: sourceInfo must be a dictionary.")

                        missingFields = [field for field in identityFields if field not in sourceInfo]
                        if missingFields:
                            raise ValueError(f"{label}: missing source-image identity fields {missingFields}.")

                        prefix = sourceInfo["filesystemPrefix"]
                        filename = sourceInfo["filename"]
                        datasetPath = sourceInfo["datasetPath"]
                        imageAxis = sourceInfo["imageAxis"]
                        imageIndex = sourceInfo["imageIndex"]
                        reportedIndex = info.imageIndices[groupIndex]

                        if not isinstance(prefix, str):
                            raise ValueError(f"{label}: filesystemPrefix must be a string.")
                        if not isinstance(filename, str) or not filename.strip():
                            raise ValueError(f"{label}: a source filename is required.")
                        if not isinstance(datasetPath, str) or not datasetPath.strip():
                            raise ValueError(f"{label}: a source datasetPath is required.")
                        if not validIndex(imageAxis) or not validIndex(imageIndex):
                            raise ValueError(f"{label}: imageAxis and imageIndex must be nonnegative integers.")
                        if not validIndex(reportedIndex) or int(reportedIndex) != int(imageIndex):
                            raise ValueError(f"{label}: imageIndices disagrees with sourceInfo.imageIndex.")

                        # Match exact recorded source references, not names or local IDs.
                        # No path normalization, source-file reads, or content hashing.
                        sourceKey = (prefix, filename, datasetPath, int(imageAxis), int(imageIndex))
                        identity = dict(zip(identityFields, sourceKey))

                        if stimulusID in stimulusSources and stimulusSources[stimulusID] != sourceKey:
                            raise ValueError(f"{label}: stimulusId {stimulusID!r} identifies multiple source images.")
                        stimulusSources[stimulusID] = sourceKey

                        name = str(info.groupNames[groupIndex])
                        if sourceKey not in conditionColumns:
                            column = len(conditions)
                            conditionColumns[sourceKey] = column
                            conditions.append({
                                "conditionIndex": column,
                                "identity": identity,
                                "name": name,
                                "names": [],
                                "stimulusIDs": [],
                            })
                        column = conditionColumns[sourceKey]
                        condition = conditions[column]

                        # Preserve aliases without treating either names or IDs as keys.
                        if name not in condition["names"]: condition["names"].append(name)
                        if stimulusID not in condition["stimulusIDs"]: condition["stimulusIDs"].append(stimulusID)

                        for trialNum, volume, timestamp, startTime in zip(trialNums, volumes, timestamps, startTimes):
                            if not validIndex(trialNum) or int(trialNum) in seenTrials:
                                raise ValueError("Trial indices must be unique nonnegative integers within a run.")
                            trialNum = int(trialNum)
                            seenTrials.add(trialNum)

                            if not validIndex(volume) or int(volume) not in volumeRows:
                                raise ValueError(
                                    f"Run {runIndex}, trial {trialNum}: onset volume "
                                    f"{volume!r} is unavailable in the selected MRI data."
                                )
                            if not finiteNumber(timestamp) or not finiteNumber(startTime):
                                raise ValueError("Trial onset times must be finite.")

                            originalVolume = int(volume)
                            rows.append({
                                "fitRunIndex": fitRunIndex,
                                "scanIndex": scanIndex,
                                "runIndex": runIndex,
                                "scanUUID": scan.uuid,
                                "trialNum": trialNum,
                                "groupIndex": groupIndex,
                                "conditionIndex": column,
                                "stimulusID": stimulusID,
                                "stimulusName": name,
                                "sourceInfo": deepcopy(sourceInfo),
                                "imageIndex": int(imageIndex),
                                "volumeRow": volumeRows[originalVolume],
                                "originalVolumeIndex": originalVolume,
                                "onsetTimestamp": float(timestamp),
                                "taskRelativeOnsetSeconds": float(startTime),
                            })

                    if not rows:
                        raise ValueError(f"Run {runIndex}: no nonblank presentations are available.")
                    if len(rows) != info.nTrialsTotal:
                        raise ValueError("Included trial count disagrees with grouped presentations.")

                    # GLMsingle reports trials chronologically, not in the
                    # stimulus inventory/group order returned by getTrialsInfo().
                    rows.sort(key=lambda row: row["volumeRow"])
                    if len({row["volumeRow"] for row in rows}) != len(rows):
                        raise ValueError(f"Run {runIndex}: multiple presentations map to the same volume row.")
                    if any(
                        right["onsetTimestamp"] < left["onsetTimestamp"]
                        for left, right in zip(rows, rows[1:])
                    ):
                        raise ValueError("Recorded onset order disagrees with design-volume order.")

                    for row in rows:
                        row["betaIndex"] = len(allTrials)
                        allTrials.append(row)
                    trialsByRun.append(rows)

                    runRecords.append({
                        "fitRunIndex": fitRunIndex,
                        "scanIndex": scanIndex,
                        "runIndex": runIndex,
                        "scanUUID": scan.uuid,
                        "taskName": info.taskName,
                        "taskSaveName": info.taskSaveName,
                        "segmentNum": info.segmentNum,
                        "direction": info.direction,
                        "nVolumes": int(data.shape[1]),
                        "nTrials": len(rows),
                        "trSeconds": float(tr),
                        "volumeIndices": volumeIndices,
                        "spatialShape": shape,
                        "affine": deepcopy(mriMetadata["affine"]),
                        "spatialUnits": deepcopy(mriMetadata["spatialUnits"]),
                        "input": deepcopy(matrix.provenance),
                        "omittedTrialNums": deepcopy(info.omittedTrialNums),
                    })

                # Use the union of observed conditions. Conditions absent from
                # an individual run retain all-zero columns in that run.
                designs = []
                for record, rows in zip(runRecords, trialsByRun):
                    design = np.zeros((record["nVolumes"], len(conditions)), dtype=np.float32)
                    for row in rows:
                        design[row["volumeRow"], row["conditionIndex"]] = 1.0
                    designs.append(design)

                # Report the actual prepared inputs, without reading MRI voxel data.
                presentationsByRun = np.stack([
                    design.sum(axis=0).astype(np.int64)
                    for design in designs
                ])
                totalPresentations = presentationsByRun.sum(axis=0)
                runsPerCondition = (presentationsByRun > 0).sum(axis=0)
                repeatCounts, stimulusCounts = np.unique(totalPresentations, return_counts=True)
                repeatSummary = ", ".join(
                    f"{nStimuli} stimuli {nRepeats}x"
                    for nRepeats, nStimuli in zip(repeatCounts, stimulusCounts)
                )
                volumeCounts = [record["nVolumes"] for record in runRecords]
                presentationCounts = [record["nTrials"] for record in runRecords]
                onsetLabel = "trial START" if segmentNum is None else f"segment {segmentNum} (zero-based)"

                pglMessages.message(
                    f"GLMsingle inputs: {len(runRecords)} runs; "
                    f"{len(conditions)} stimuli; {len(allTrials)} presentations "
                    f"({repeatSummary})"
                )
                pglMessages.message(
                    f"Conditions shared across runs: {int((runsPerCondition > 1).sum())}; "
                    f"identity: recorded source image"
                )
                pglMessages.message(
                    f"TR: {commonTR:.6g} s; stimulus duration: {float(stimDur):.6g} s; "
                    f"onsets: {onsetLabel}"
                )
                pglMessages.message(
                    f"Voxels per run: {nVoxels}; spatial shape: {tuple(commonShape)}; "
                    f"volumes per run: {volumeCounts}"
                )
                pglMessages.message(f"Presentations per run: {presentationCounts}")

                return {
                    "design": designs,
                    "matrices": matrices,
                    "conditions": conditions,
                    "trials": allTrials,
                    "runs": runRecords,
                    "stimDur": float(stimDur),
                    "tr": commonTR,
                    "nVoxels": nVoxels,
                    "nTrials": len(allTrials),
                    "conditionIdentityMethod": "recordedSourceImage",
                }

            except BaseException:
                for matrix in matrices:
                    matrix.close()
                raise
        
        def _matrixLabel(self, role):
            """Return a readable label without changing native result keys."""
            labels = {
                "typea.betasmd": "typeA_onoffBetas",
                "typea.onoffR2": "typeA_onoffR2",
                "typea.meanvol": "typeA_meanVolume",
                "typeb.betasmd": "typeB_betas",
                "typeb.R2": "typeB_R2",
                "typeb.R2run": "typeB_R2ByRun",
                "typeb.meanvol": "typeB_meanVolume",
                "typeb.FitHRFR2": "typeB_hrfCandidateR2",
                "typeb.FitHRFR2run": "typeB_hrfCandidateR2ByRun",
                "typeb.HRFindex": "typeB_hrfIndex",
                "typeb.HRFindexrun": "typeB_hrfIndexByRun",
            }
            return labels.get(role, role.replace(".", "_"))

        def _convertTypeBResults(self, prepared, results, designInfo, *, artifactDirectory=None):
            """Validate result associations and construct unattached analysis matrices.

            Supports the currently implemented ordinary Type B, multi-HRF-library path.
            Preserves native numerical outputs, including NaN and infinity.

            Validates explicit dimensions and chronological trial/design associations.
            Does not impose numerical fit-quality thresholds or recompute HRF selection.

            Additional returned fields/models are preserved in a native file artifact.
            They are not silently discarded or passed to Zarr as object arrays.

            Native HRF indices are preserved. The inspected multi-HRF branch of the
            installed implementation uses zero-based indices.

            Does not attach matrices, register products, or replace MRI images.
            The caller owns the returned wrappers and closes them after use.
            """
            from copy import deepcopy
            from pathlib import Path

            import numpy as np
            import xarray as xr

            from .pglData import pglDataMatrix

            nVoxels = prepared["nVoxels"]
            nTrials = prepared["nTrials"]
            nRuns = len(prepared["runs"])
            nConditions = len(prepared["conditions"])
            params = designInfo["params"]

            if (
                params["wantlibrary"] != 1
                or params["wantglmdenoise"] != 0
                or params["wantfracridge"] != 0
                or params.get("wantlss", 0) != 0
            ):
                raise ValueError("This converter currently implements ordinary Type B without denoising/ridge.")

            hrfLibrary = np.asarray(params["hrflibrary"])
            if hrfLibrary.ndim != 2 or hrfLibrary.shape[1] <= 1:
                raise ValueError("Single-HRF/assumed-HRF conversion is not implemented yet.")
            nHRFs = hrfLibrary.shape[1]

            if not np.isclose(designInfo["tr"], prepared["tr"], rtol=0, atol=1e-8):
                raise ValueError("Saved GLMsingle TR differs from the prepared TR.")
            if not np.isclose(designInfo["stimdur"], prepared["stimDur"], rtol=0, atol=1e-8):
                raise ValueError("Saved stimulus duration differs from preparation.")

            trials = prepared["trials"]
            if len(trials) != nTrials:
                raise ValueError("Prepared trial count is inconsistent.")
            if [trial["betaIndex"] for trial in trials] != list(range(nTrials)):
                raise ValueError("Prepared beta indices must be contiguous and zero-based.")

            trialOrder = [(trial["fitRunIndex"], trial["volumeRow"]) for trial in trials]
            if trialOrder != sorted(trialOrder) or len(set(trialOrder)) != nTrials:
                raise ValueError("Prepared trial mapping is not uniquely chronological.")
            if any(
                not 0 <= trial["fitRunIndex"] < nRuns
                or not 0 <= trial["conditionIndex"] < nConditions
                for trial in trials
            ):
                raise ValueError("Prepared trial mapping contains invalid run/condition indices.")

            savedDesigns = designInfo["design"]
            singleDesigns = designInfo["designSINGLE"]
            if not (
                len(prepared["runs"]) == len(prepared["design"])
                == len(savedDesigns) == len(singleDesigns) == nRuns
            ):
                raise ValueError("Design/run counts differ from preparation.")

            expectedOrder = np.asarray([trial["conditionIndex"] for trial in trials])
            if not np.array_equal(np.asarray(designInfo["stimorder"]), expectedOrder):
                raise ValueError("GLMsingle stimorder differs from the prepared condition ordering.")

            trialCounts = []
            for fitRunIndex, (record, design, savedDesign, singleDesign) in enumerate(zip(
                prepared["runs"], prepared["design"], savedDesigns, singleDesigns,
            )):
                if record["fitRunIndex"] != fitRunIndex:
                    raise ValueError("Prepared run ordering is inconsistent.")
                if design.shape != (record["nVolumes"], nConditions):
                    raise ValueError("Prepared design shape differs from run metadata.")
                if not np.array_equal(design, savedDesign):
                    raise ValueError(f"Fit run {fitRunIndex}: saved design differs from preparation.")

                runTrials = [trial for trial in trials if trial["fitRunIndex"] == fitRunIndex]
                trialCounts.append(len(runTrials))
                if len(runTrials) != record["nTrials"]:
                    raise ValueError("Prepared run trial count is inconsistent.")

                singleDesign = np.asarray(singleDesign)
                if singleDesign.shape != (record["nVolumes"], nTrials):
                    raise ValueError("Unexpected GLMsingle single-trial design shape.")

                # Verify exact onset/column locations without allocating a second design.
                actualRows, actualColumns = np.nonzero(singleDesign)
                expectedRows = np.asarray([trial["volumeRow"] for trial in runTrials], dtype=int)
                expectedColumns = np.asarray([trial["betaIndex"] for trial in runTrials], dtype=int)

                if (
                    not np.array_equal(actualRows, expectedRows)
                    or not np.array_equal(actualColumns, expectedColumns)
                    or not np.all(singleDesign[actualRows, actualColumns] == 1)
                ):
                    raise ValueError("GLMsingle single-trial columns differ from the prepared beta mapping.")

                for trial in runTrials:
                    if not 0 <= trial["volumeRow"] < design.shape[0]:
                        raise ValueError("Prepared onset row is outside its run.")
                    if design[trial["volumeRow"], trial["conditionIndex"]] != 1:
                        raise ValueError("Prepared trial mapping differs from its onset design.")

            presentationsByRun = np.stack([design.sum(axis=0).astype(np.int64) for design in prepared["design"]])
            for field, expected in (
                ("numtrialrun", trialCounts),
                ("condcounts", presentationsByRun.sum(axis=0)),
                ("condinruns", (presentationsByRun > 0).sum(axis=0)),
            ):
                if not np.array_equal(np.asarray(designInfo[field]), np.asarray(expected)):
                    raise ValueError(f"Saved GLMsingle {field} differs from preparation.")

            if not isinstance(results, dict) or "typeb" not in results:
                raise ValueError("GLMsingle did not return the requested Type B results.")

            # Each entry specifies native shape and explicit analysis dimensions.
            specifications = {
                "typeb": {
                    "betasmd": ((nVoxels, 1, 1, nTrials), ("voxel", "trial")),
                    "R2": ((nVoxels, 1, 1), ("voxel",)),
                    "R2run": ((nVoxels, 1, 1, nRuns), ("voxel", "fitRun")),
                    "meanvol": ((nVoxels, 1, 1), ("voxel",)),
                    "FitHRFR2": ((nVoxels, 1, 1, nHRFs), ("voxel", "hrf")),
                    "FitHRFR2run": ((nVoxels, 1, 1, nRuns, nHRFs), ("voxel", "fitRun", "hrf")),
                    "HRFindex": ((nVoxels, 1, 1), ("voxel",)),
                    "HRFindexrun": ((nVoxels, 1, 1, nRuns), ("voxel", "fitRun")),
                },
                "typea": {
                    "betasmd": ((nVoxels, 1, 1, 1), ("voxel", "onoff")),
                    "onoffR2": ((nVoxels, 1, 1), ("voxel",)),
                    "meanvol": ((nVoxels, 1, 1), ("voxel",)),
                },
            }

            additionalOutputs = {}
            for modelName, modelResults in results.items():
                if modelName not in specifications:
                    additionalOutputs[modelName] = modelResults
                    continue

                if not isinstance(modelResults, dict):
                    raise TypeError(f"{modelName}: expected a result dictionary.")

                missingFields = set(specifications[modelName]) - set(modelResults)
                if missingFields:
                    raise ValueError(f"{modelName}: missing required fields {sorted(missingFields)}.")

                for field, (expectedShape, dimensions) in specifications[modelName].items():
                    values = np.asarray(modelResults[field])
                    if values.shape != expectedShape or values.dtype.kind not in "fiu":
                        raise ValueError(f"{modelName}.{field}: unexpected shape or numerical dtype.")

                extras = {field: value for field, value in modelResults.items() if field not in specifications[modelName]}
                if extras: additionalOutputs[modelName] = extras

            if additionalOutputs:
                if artifactDirectory is None:
                    raise ValueError("Additional GLMsingle outputs require artifactDirectory for preservation.")

                artifactPath = Path(artifactDirectory) / "PGL_ADDITIONAL_OUTPUTS.npy"
                # Exclusive creation prevents replacing an existing artifact.
                with artifactPath.open("xb") as stream:
                    np.save(stream, additionalOutputs, allow_pickle=True)

                pglMessages.message(f"Additional GLMsingle outputs preserved as a native artifact: {artifactPath}")

            sharedProvenance = {
                "operation": "GLMsingleAnalysis",
                "conditionIdentityMethod": prepared["conditionIdentityMethod"],
                "voxelOrder": "C",
                "voxelMapping": "voxel = ((x * ny) + y) * nz + z",
                "runs": deepcopy(prepared["runs"]),
                "trialOrder": "inputRunThenOnsetVolume",
                "hrfIndexBase": 0,
                "hrfIndexConvention": "inspectedMultiHRFLibraryBranch",
                "stimDur": float(prepared["stimDur"]),
                "tr": float(prepared["tr"]),
                "nonfiniteOutputPolicy": "preserveNativeValues",
            }

            matrices = {}
            try:
                for modelName, modelResults in results.items():
                    if modelName not in specifications: continue

                    for field, (expectedShape, dimensions) in specifications[modelName].items():
                        native = np.asarray(modelResults[field])

                        # Remove only validated singleton spatial axes.
                        # No filtering, replacement, or fit-quality threshold is applied.
                        values = native[:, 0, 0, ...]
                        role = f"{modelName}.{field}"
                        nonfiniteCount = int(np.count_nonzero(~np.isfinite(values)))

                        data = xr.DataArray(
                            values,
                            dims=dimensions,
                            coords={dimension: np.arange(size, dtype=np.int64) for dimension, size in zip(dimensions, values.shape)},
                            attrs={
                                "analysis": "GLMsingle",
                                "model": modelName,
                                "field": field,
                                "nativeShape": list(expectedShape),
                                "nonfiniteCount": nonfiniteCount,
                                "nonfiniteOutputPolicy": "preserveNativeValues",
                            },
                        )

                        matrix = pglDataMatrix(data, name=f"{self.outputName}_{self._matrixLabel(role)}")
                        matrices[role] = matrix
                        matrix.provenance = {
                            **deepcopy(sharedProvenance),
                            "model": modelName,
                            "field": field,
                            "nativeShape": list(expectedShape),
                            "dimensions": list(dimensions),
                            "nonfiniteCount": nonfiniteCount,
                        }

                        if nonfiniteCount:
                            pglMessages.message(f"GLMsingle {role}: retaining {nonfiniteCount} nonfinite output values without replacement.")

                pglMessages.message(
                    f"Validated GLMsingle Type B dimensions and mappings: {nVoxels} voxels; "
                    f"{nTrials} chronological trial betas; {nRuns} runs; {nHRFs} HRF candidates."
                )
                return matrices

            except BaseException:
                for matrix in matrices.values(): matrix.close()
                raise
        
        def _materializeFitData(self, prepared):
            """Load independent writable float32 arrays for GLMsingle.

            Preserves the prepared voxel/volume order.
            Does not normalize, detrend, resample, or replace nonfinite values.

            GLMsingle may modify its input arrays or list entries. These allocations
            must therefore not share storage with the session's MRI arrays.

            The caller retains ownership of prepared lazy views and closes them.
            """
            import dask
            import numpy as np

            progress = currentProgress()
            matrices = prepared["matrices"]
            runs = prepared["runs"]

            if not matrices or len(matrices) != len(runs):
                raise ValueError("Prepared MRI matrix/run counts are inconsistent.")

            inputBytes = sum(
                prepared["nVoxels"] * record["nVolumes"] * np.dtype(np.float32).itemsize
                for record in runs
            )
            betaBytes = prepared["nVoxels"] * prepared["nTrials"] * np.dtype(np.float32).itemsize

            pglMessages.message(
                f"GLMsingle memory: owned float32 inputs require {inputBytes / 2**30:.3f} GiB; "
                f"one single-trial beta array requires {betaBytes / 2**30:.3f} GiB.\n"
                "These are array sizes, not a peak-memory estimate. Source materialization, "
                "HRF fitting, diagnostics, and library workspaces require additional memory."
            )
            pglMessages.message(
                "Loading current MRI inputs into independent writable float32 arrays. "
                "No normalization, detrending, resampling, or nonfinite replacement is performed."
            )

            fitData = []

            for fitRunIndex, (matrix, record) in enumerate(zip(matrices, runs)):
                progress(
                    fitRunIndex / len(runs),
                    f"Loading GLMsingle input — scan {record['scanIndex']}",
                )

                source = matrix.requireData()
                expectedShape = (prepared["nVoxels"], record["nVolumes"])

                if tuple(source.dims) != ("voxel", "volume") or source.shape != expectedShape:
                    raise ValueError("Prepared MRI matrix dimensions changed before loading.")
                if source.dtype.kind not in "fiu":
                    raise ValueError("GLMsingle requires real numerical MRI data.")

                # Compute without changing the source DataArray or installing its result
                # back into the lazy matrix. Do not introduce another progress display.
                with dask.config.set(scheduler="synchronous"):
                    with noProgressBar():
                        loaded = source.compute()

                # copy=True is essential even when the source is already float32.
                # NumPy conversion alone does not guarantee independent storage.
                with np.errstate(over="ignore", invalid="ignore"):
                    values = np.array(loaded.data, dtype=np.float32, order="C", copy=True)
                del loaded

                if values.shape != expectedShape:
                    raise ValueError("Materialized MRI dimensions differ from preparation.")
                if not values.flags.owndata or not values.flags.writeable:
                    raise ValueError("Fit input allocation is not independently owned and writable.")
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"Scan {record['scanIndex']}: nonfinite MRI values or float32 overflow. "
                        "Values will not be silently replaced."
                    )

                fitData.append(values)
                pglMessages.message(
                    f"Loaded GLMsingle input {fitRunIndex + 1}/{len(runs)}: "
                    f"scan {record['scanIndex']}; shape={values.shape}; "
                    f"source={record['input']['sourceStageID']}"
                )

            progress(1, "GLMsingle input loading complete")
            return fitData
        
        @operationProgress("Fitting GLMsingle")
        def _fitPreparedInputs(self, prepared, *, params, cacheDirectory=None):
            """Fit and validate the currently supported Type B execution path.

            All scientific options are supplied explicitly by the caller.
            This implementation requires:
                wantlibrary=1
                wantglmdenoise=0
                wantfracridge=0
                wantlss=0 or omitted

            Output routing is action-owned:
                Type A/B are returned in memory.
                Large native model files are disabled.
                Support files and figures remain in a fresh dedicated workspace.

            Does not attach matrices, register products, or change current MRI images.
            The caller closes prepared input views and returned analysis wrappers.

            Failed workspaces are retained for diagnosis. Never passes an existing
            session cache root or unrelated directory to GLMsingle.
            """
            import inspect
            import os
            import tempfile
            from copy import deepcopy
            from importlib.metadata import version
            from pathlib import Path

            import numpy as np

            from glmsingle.glmsingle import GLM_single

            progress = currentProgress()

            # Validate the installed callable interface before loading MRI arrays.
            inspect.signature(GLM_single).bind(params={})
            inspect.signature(GLM_single.fit).bind(
                None,
                [],
                [],
                stimdur=prepared["stimDur"],
                tr=prepared["tr"],
                outputdir=None,
                figuredir=None,
            )
            toolInfo = {
                "name": "GLMsingle",
                "version": version("glmsingle"),
                "implementation": str(Path(inspect.getfile(GLM_single)).resolve()),
            }

            if not isinstance(params, dict):
                raise TypeError("params must be an explicit GLMsingle options dictionary.")

            requiredOptions = {
                "wantlibrary": 1,
                "wantglmdenoise": 0,
                "wantfracridge": 0,
            }
            for name, expected in requiredOptions.items():
                value = params.get(name)
                if not isinstance(value, (int, np.integer)) or value != expected:
                    raise ValueError(
                        f"This verified Type B execution path requires explicit {name}={expected}. "
                        "Other model paths are not implemented here yet."
                    )

            wantLSS = params.get("wantlss", 0)
            if not isinstance(wantLSS, (int, np.integer)) or wantLSS != 0:
                raise ValueError("LSS is not supported by this verified Type B execution path.")

            # These are storage choices, not scientific modeling choices.
            # Reject conflicting requests instead of silently overriding them.
            outputOptions = {
                "wantfileoutputs": [0, 0, 0, 0],
                "wantmemoryoutputs": [1, 1, 0, 0],
                "wanthdf5": 0,
            }
            for name, expected in outputOptions.items():
                if name in params and not np.array_equal(np.asarray(params[name]), np.asarray(expected)):
                    raise ValueError(
                        f"{name} is managed by this action and must be {expected!r}. "
                        "Native-model retention will be handled explicitly during publication."
                    )

            requestedParams = deepcopy(params)
            fitParams = deepcopy(params)
            fitParams.update(deepcopy(outputOptions))

            # Constructing the model is cheap compared with loading the MRI inputs.
            # Constructor validation therefore belongs before materialization.
            model = GLM_single(params=deepcopy(fitParams))

            if cacheDirectory is None:
                cacheRoot = Path.home() / ".cache" / "pgl"
            else:
                cacheText = os.fspath(cacheDirectory)
                if (
                    not isinstance(cacheText, str) or not cacheText.strip()
                    or "://" in cacheText or "::" in cacheText
                    or any(character in cacheText for character in ("\x00", "\n", "\r"))
                ):
                    raise ValueError("cacheDirectory must be an ordinary local path.")
                cacheRoot = Path(cacheText).expanduser().absolute()

            workspaceRoot = cacheRoot / "glmsingle"
            workspaceRoot.mkdir(parents=True, exist_ok=True)
            workspace = Path(tempfile.mkdtemp(prefix="fit-", dir=workspaceRoot))

            # Leave these dedicated children nonexistent until GLMsingle creates them.
            outputDirectory = workspace / "outputs"
            figureDirectory = workspace / "figures"

            pglMessages.message(
                f"GLMsingle {toolInfo['version']}: {toolInfo['implementation']}\n"
                f"Workspace: {workspace}\n"
                "Model: Type B, HRF library enabled; denoising disabled; ridge disabled.\n"
                "Type A ON/OFF results are retained separately from single-trial betas."
            )
            pglMessages.message(f"GLMsingle submitted options: {fitParams}")

            fitData = None
            fitDesigns = None
            converted = None

            try:
                # Give the library disposable design copies as well as owned MRI arrays.
                fitDesigns = [
                    np.array(design, dtype=np.float32, order="C", copy=True)
                    for design in prepared["design"]
                ]
                fitData = progressCall(
                    progress, 0.02, 0.25,
                    "Loading GLMsingle MRI inputs",
                    self._materializeFitData,
                    prepared,
                )

                # No artificial within-fit percentages: library output remains live
                # under the action infrastructure's stdout/stderr capture.
                results = progressCall(
                    progress, 0.25, 0.85,
                    "Fitting GLMsingle — library output below",
                    model.fit,
                    fitDesigns,
                    fitData,
                    stimdur=prepared["stimDur"],
                    tr=prepared["tr"],
                    outputdir=str(outputDirectory),
                    figuredir=str(figureDirectory),
                )

                # Release our input references before converting/storing large outputs.
                # The external model may retain its own references until this method exits.
                fitData = None
                fitDesigns = None

                designInfoPath = outputDirectory / "DESIGNINFO.npy"
                if not designInfoPath.is_file():
                    raise ValueError("GLMsingle did not produce the required DESIGNINFO.npy.")

                # Trusted artifact generated by this invocation, not an arbitrary file.
                saved = np.load(designInfoPath, allow_pickle=True)
                if saved.shape != () or saved.dtype != object:
                    raise ValueError("Unexpected DESIGNINFO.npy representation.")
                designInfo = saved.item()
                if not isinstance(designInfo, dict) or not isinstance(designInfo.get("params"), dict):
                    raise ValueError("GLMsingle design information lacks its parameter snapshot.")

                converted = progressCall(progress, 0.85, 0.98, "Validating GLMsingle results and trial mapping", self._convertTypeBResults, prepared, results, designInfo, artifactDirectory=outputDirectory)
                
                # Report resolved values, not presumed defaults. DESIGNINFO records the
                # library's parameter snapshot when it writes that support artifact.
                savedParams = designInfo["params"]
                pglMessages.message(
                    "GLMsingle saved parameter snapshot:\n"
                    f"  Percent-BOLD output: {savedParams.get('wantpercentbold')}\n"
                    f"  Polynomial degrees per run: {savedParams.get('maxpolydeg')}\n"
                    f"  HRF library shape: {np.asarray(savedParams['hrflibrary']).shape}\n"
                    f"  Seed: {savedParams.get('seed')}\n"
                    f"  Session labels (external one-based convention): "
                    f"{np.asarray(savedParams.get('sessionindicator')).tolist()}\n"
                    "  Denoising/ridge hyperparameter CV: disabled for this explicit Type B fit."
                )

                artifacts = [
                    {
                        "relativePath": path.relative_to(workspace).as_posix(),
                        "localPath": str(path),
                        "sizeBytes": path.stat().st_size,
                    }
                    for path in sorted(workspace.rglob("*"))
                    if path.is_file()
                ]
                pglMessages.message(
                    f"GLMsingle fit validated: {len(converted)} analysis matrices; "
                    f"{len(artifacts)} support/figure files. Nothing has been published yet."
                )

                return {
                    "matrices": converted,
                    "designInfo": designInfo,
                    "requestedParams": requestedParams,
                    "submittedParams": fitParams,
                    "tool": toolInfo,
                    "workspace": str(workspace),
                    "artifacts": artifacts,
                    "computedModels": ["typea", "typeb"],
                }

            except BaseException:
                if converted is not None:
                    for matrix in converted.values():
                        matrix.close()
                pglMessages.message(f"GLMsingle failed; diagnostic workspace retained at {workspace}")
                raise

            finally:
                fitData = None
                fitDesigns = None

        @operationProgress("Storing GLMsingle analysis")
        def _publishResults(self, session, prepared, fitted, *, outputName, retainRoles):
            """Persist validated GLMsingle analysis matrices and register their product.

            retainRoles:
                Explicit nonempty list of converted matrix roles to retain.
                The final Type B single-trial betas must be included.

            Always retains:
                Prepared design matrices.
                Trial, condition, run, geometry, and input provenance mappings.
                Requested/submitted options and the saved library parameter snapshot.
                Generated support files and figures.

            Large model matrices omitted by retainRoles are explicitly recorded.
            Native support files may contain nested structures and remain file artifacts.

            All matrices are written and reopened before attachment. Scientific
            commitment remains the action runner's responsibility.

            Failure may leave cache files or diagnostic attachments. No filesystem or
            observer rollback is claimed. Failed actions are not safe to checkpoint.

            Never publishes image replacements or changes current-image selection.
            """
            import json
            from copy import deepcopy
            from pathlib import Path
            from uuid import uuid4

            import numpy as np
            import xarray as xr

            from .pglData import pglDataMatrix
            from .pglExternalTools import getExecutionRecorder, pglExternalTool
            from .pglStorage import pglStorage

            progress = currentProgress()
            recorder = getExecutionRecorder()

            if (
                not getattr(self, "_executionInProgress", False)
                or recorder is None
                or not self.actionHistory
                or recorder.actionID != self.actionHistory[-1].actionID
            ):
                raise RuntimeError("GLMsingle publication requires its producing action's active execution.")
            if session.mri is None or session.mri._getSession() is not session:
                raise ValueError("GLMsingle publication requires MRI ownership by this session.")
            if session._failedActionIDs or session.mri._failedActionIDs:
                raise ValueError("Cannot publish on a working object with unresolved failures.")

            if (
                not isinstance(outputName, str) or not outputName.strip()
                or any(character in outputName for character in ("\x00", "\n", "\r"))
            ):
                raise ValueError("outputName must be nonblank without NUL or newlines.")

            if any(
                record.get("name") == outputName and record.get("recordStatus") == "committed"
                for entry in session.actionHistory
                for record in entry.actionProducts
            ):
                raise ValueError(f"Product {outputName!r} already exists.")
            if any(record["name"] == outputName for record in recorder.exportProducts()):
                raise ValueError("This execution already registered the requested outputName.")

            available = fitted["matrices"]
            if (
                not isinstance(retainRoles, (list, tuple)) or not retainRoles
                or any(not isinstance(role, str) for role in retainRoles)
                or len(set(retainRoles)) != len(retainRoles)
            ):
                raise ValueError("retainRoles must be a nonempty sequence of unique matrix-role strings.")
            if set(retainRoles) - set(available):
                raise ValueError(f"Unknown retained matrix roles: {sorted(set(retainRoles) - set(available))}")
            if "typeb.betasmd" not in retainRoles:
                raise ValueError("The final Type B single-trial betas must be retained.")

            omittedRoles = sorted(set(available) - set(retainRoles))
            workspace = Path(fitted["workspace"]).resolve()
            if not workspace.is_dir():
                raise ValueError("The GLMsingle workspace is unavailable.")

            # A separate publication directory must not already contain another attempt.
            publicationDirectory = workspace / f"publication-{uuid4()}"
            publicationDirectory.mkdir()
            matrixRoot = publicationDirectory / "matrices"

            productID = str(uuid4())
            executionID = recorder.actionID
            mri = session.mri
            attachments = dict(mri.dataMatrices)

            def jsonValue(value):
                """Encode supported metadata explicitly; never stringify unknown objects."""
                if isinstance(value, np.ndarray): return jsonValue(value.tolist())
                if isinstance(value, np.generic): return jsonValue(value.item())
                if value is None or isinstance(value, (str, bool, int)): return value
                if isinstance(value, float):
                    if not np.isfinite(value):
                        raise ValueError("Nonfinite metadata requires an explicit encoding.")
                    return value
                if isinstance(value, (list, tuple)):
                    return [jsonValue(item) for item in value]
                if isinstance(value, dict):
                    if any(not isinstance(key, str) for key in value):
                        raise TypeError("Metadata dictionary keys must be strings.")
                    return {key: jsonValue(item) for key, item in value.items()}
                raise TypeError(f"Unsupported metadata type: {type(value).__name__}")

            # Snapshot metadata before writing. No scientific mapping is deferred to QC.
            manifest = jsonValue({
                "schemaVersion": 1,
                "operation": "GLMsingleAnalysis",
                "productID": productID,
                "producerActionID": executionID,
                "tool": fitted["tool"],
                "conditionIdentityMethod": prepared["conditionIdentityMethod"],
                "runs": prepared["runs"],
                "conditions": prepared["conditions"],
                "trials": prepared["trials"],
                "stimDur": prepared["stimDur"],
                "tr": prepared["tr"],
                "nVoxels": prepared["nVoxels"],
                "nTrials": prepared["nTrials"],
                "voxelOrder": "C",
                "voxelMapping": "voxel = ((x * ny) + y) * nz + z",
                "geometryPolicy": "Per-run recorded geometry; anatomical alignment is an upstream responsibility.",
                "requestedParams": fitted["requestedParams"],
                "submittedParams": fitted["submittedParams"],
                "savedLibraryParams": fitted["designInfo"]["params"],
                "parameterSnapshotSource": "GLMsingle DESIGNINFO.npy",
                "computedModels": fitted["computedModels"],
                "retainedModelMatrixRoles": list(retainRoles),
                "omittedModelMatrixRoles": omittedRoles,
                "hrfIndexBase": 0,
                "hrfIndexConvention": "inspectedMultiHRFLibraryBranch",
            })

            dependencies = list(dict.fromkeys(
                record["input"]["productID"]
                for record in prepared["runs"]
                if record["input"]["productID"] is not None
            ))

            # Derived inputs were consumed by getDataMatrix during preparation.
            # Raw inputs remain explicit acquisition-stage records in the manifest.
            for identifier in dependencies:
                product = pglProductSelector(productID=identifier, kind="image").resolve(session)
                if product is None:
                    raise ValueError("A consumed image product is no longer available.")

            # Account for and verify every generated support/figure artifact.
            supportFiles = []
            for artifact in fitted["artifacts"]:
                path = Path(artifact["localPath"]).resolve()
                if not path.is_relative_to(workspace) or not path.is_file():
                    raise ValueError("A generated artifact is missing or outside the action workspace.")
                if path.relative_to(workspace).as_posix() != artifact["relativePath"]:
                    raise ValueError("Artifact relative path differs from its recorded path.")
                identity = pglExternalTool._fileIdentity(path)
                if identity["size_bytes"] != artifact["sizeBytes"]:
                    raise ValueError("A generated artifact changed size before publication.")
                supportFiles.append((artifact["relativePath"], path, identity))

            # Meaningful filenames, not support.0/support.1.
            supportArtifacts = []
            usedRoles = {"analysis.json"}
            for relativePath, path, identity in supportFiles:
                extension = path.suffix
                baseName = path.stem
                role = path.name
                suffix = 2
                while role in usedRoles:
                    role = f"{baseName}_v{suffix}{extension}"
                    suffix += 1
                usedRoles.add(role)
                supportArtifacts.append({"role": role, "relativePath": relativePath})

            def requireUnchanged():
                """Check ownership, attachments, and exact current input selections."""
                if session.mri is not mri or mri._getSession() is not session:
                    raise ValueError("MRI ownership changed during publication.")
                if session._failedActionIDs or mri._failedActionIDs:
                    raise ValueError("The working object acquired an unresolved failure.")
                if (
                    set(mri.dataMatrices) != set(attachments)
                    or any(mri.dataMatrices[name] is not matrix for name, matrix in attachments.items())
                ):
                    raise ValueError("MRI matrix attachments changed during publication.")

                for record in prepared["runs"]:
                    view = mri.getDataMatrix(scanNum=record["scanIndex"])
                    if view is None:
                        raise ValueError("A prepared MRI input is no longer available.")
                    try:
                        if jsonValue(view.provenance) != jsonValue(record["input"]):
                            raise ValueError("Current MRI input selection changed during the analysis.")
                    finally:
                        view.close()

            opened = {}
            temporaryMatrices = []
            declarations = []
            attachmentAttempted = False

            try:
                outputs = [(role, available[role]) for role in retainRoles]

                # Designs have ordinary numeric dimensions, not object-valued cells.
                for record, design in zip(prepared["runs"], prepared["design"]):
                    role = f"design_scan{record['scanIndex']:02d}_run{record['runIndex']:02d}"
                    data = xr.DataArray(
                        design,
                        dims=("volume", "condition"),
                        coords={
                            "volume": np.arange(design.shape[0], dtype=np.int64),
                            "condition": np.arange(design.shape[1], dtype=np.int64),
                            "originalVolumeIndex": (
                                "volume", np.asarray(record["volumeIndices"], dtype=np.int64),
                            ),
                        },
                        attrs={
                            "analysis": "GLMsingle",
                            "fitRunIndex": record["fitRunIndex"],
                            "scanIndex": record["scanIndex"],
                            "runIndex": record["runIndex"],
                            "trSeconds": prepared["tr"],
                            "stimDurSeconds": prepared["stimDur"],
                        },
                    )
                    matrix = pglDataMatrix(data, name=role)
                    temporaryMatrices.append(matrix)
                    matrix.provenance = {
                        "operation": "GLMsingleDesign",
                        "input": deepcopy(record["input"]),
                        "conditionIdentityMethod": prepared["conditionIdentityMethod"],
                    }
                    outputs.append((role, matrix))

                requireUnchanged()

                for position, (role, matrix) in enumerate(outputs):
                    baseName = f"{outputName}_{self._matrixLabel(role)}"
                    matrixName = baseName
                    suffix = 2
                    while matrixName in attachments or matrixName in opened:
                        matrixName = f"{baseName}_v{suffix}"
                        suffix += 1
                    source = matrix.requireData()

                    # Create an independent wrapper without copying the large array.
                    data = source.copy(deep=False)
                    data.attrs = deepcopy(source.attrs)
                    output = pglDataMatrix(data, name=matrixName)
                    temporaryMatrices.append(output)
                    output.provenance = jsonValue({
                        **deepcopy(matrix.provenance),
                        "productID": productID,
                        "producerActionID": executionID,
                        "outputRole": role,
                    })
                    pglStorage.requireSafeSave(output)

                    start = 0.85 * position / len(outputs)
                    stop = 0.85 * (position + 1) / len(outputs)
                    with noProgressBar():
                        savedPath = progressCall(
                            progress, start, stop,
                            f"Writing GLMsingle {role}",
                            pglStorage.saveDataMatrix,
                            output,
                            matrixRoot,
                        )

                    restored = pglStorage.loadDataMatrix(savedPath)
                    opened[matrixName] = restored
                    stored = restored.requireData()

                    if restored.uuid != output.uuid:
                        raise ValueError("Zarr restoration changed an analysis matrix UUID.")
                    if (
                        stored.shape != data.shape
                        or stored.dtype != data.dtype
                        or tuple(stored.dims) != tuple(data.dims)
                        or jsonValue(stored.attrs) != jsonValue(data.attrs)
                    ):
                        raise ValueError("Zarr restoration changed analysis dimensions, dtype, or metadata.")

                    if set(stored.coords) != set(data.coords):
                        raise ValueError("Zarr restoration changed analysis coordinate names.")
                    for name in data.coords:
                        if (
                            stored.coords[name].dims != data.coords[name].dims
                            or not np.array_equal(stored.coords[name].values, data.coords[name].values)
                        ):
                            raise ValueError(f"Zarr restoration changed coordinate {name!r}.")

                    declarations.append({
                        "role": role,
                        "matrixName": matrixName,
                        "matrixUUID": restored.uuid,
                        # Publication provenance, not a promise that checkpoint storage
                        # will preserve the same storage revision.
                        "publishedMatrixRevisionID": restored.matrixRevisionID,
                        "dimensions": list(stored.dims),
                        "shape": list(stored.shape),
                        "dtype": str(stored.dtype),
                    })

                manifest["matrices"] = jsonValue(declarations)
                manifest["supportFiles"] = [
                    {"relativePath": relativePath, **identity}
                    for relativePath, path, identity in supportFiles
                ]
                manifestPath = publicationDirectory / "analysis.json"
                manifestPath.write_text(
                    json.dumps(manifest, indent=2, allow_nan=False),
                    encoding="utf-8",
                )
                manifestIdentity = pglExternalTool._fileIdentity(manifestPath)

                requireUnchanged()
                if set(opened).intersection(attachments):
                    raise ValueError("A GLMsingle matrix attachment name already exists.")

                files = {}
                filesToAttach = [
                    ("analysis.json", manifestPath, manifestIdentity),
                    *[(artifact["role"], path, identity) for artifact, (_, path, identity) in zip(supportArtifacts, supportFiles)],
                ]

                progress(0.9, "Attaching GLMsingle manifest, support files, and figures")
                for role, path, identity in filesToAttach:
                    if pglExternalTool._fileIdentity(path) != identity:
                        raise ValueError("A GLMsingle artifact changed during publication.")

                    # Extensions stay at the end, including versioned names.
                    rolePath = Path(role)
                    baseName = f"{outputName}_{rolePath.stem}"
                    extension = rolePath.suffix
                    attachmentName = f"{baseName}{extension}"
                    suffix = 2
                    while attachmentName in session.derivedFiles:
                        attachmentName = f"{baseName}_v{suffix}{extension}"
                        suffix += 1
                    mediaType = {
                        ".json": "application/json",
                        ".png": "image/png",
                        ".pdf": "application/pdf",
                    }.get(path.suffix.lower(), "application/octet-stream")

                    session.addDerivedFile(attachmentName, path, mediaType=mediaType)
                    attachment = session.derivedFiles[attachmentName]
                    if any(attachment[key] != identity[key] for key in ("sha256", "size_bytes")):
                        raise ValueError("Attached GLMsingle artifact has a different identity.")
                    files[role] = attachmentName

                requireUnchanged()

                # One assignment attaches the complete matrix batch. If an observer
                # raises after assignment, do not close potentially attached readers.
                attachmentAttempted = True
                mri.dataMatrices = {**attachments, **opened}

                product = self.registerProduct(
                    outputName,
                    "GLMsingleAnalysis",
                    session=session,
                    productID=productID,
                    files=files,
                    inputProductIDs=dependencies,
                    metadata=jsonValue({
                        "schemaVersion": 1,
                        "operation": "GLMsingleAnalysis",
                        "tool": fitted["tool"],
                        "matrices": declarations,
                        "manifestRole": "analysis.json",
                        "supportArtifacts": supportArtifacts,
                        "inputs": [record["input"] for record in prepared["runs"]],
                        "computedModels": fitted["computedModels"],
                        "retainedModelMatrixRoles": list(retainRoles),
                        "omittedModelMatrixRoles": omittedRoles,
                        "nRuns": len(prepared["runs"]),
                        "nConditions": len(prepared["conditions"]),
                        "nTrials": prepared["nTrials"],
                        "conditionIdentityMethod": prepared["conditionIdentityMethod"],
                        "outputStorage": "pgl.matrix.zarr.v1",
                    }),
                )

                progress(1, "GLMsingle analysis registered; awaiting action commitment")
                pglMessages.message(
                    f"Registered GLMsingle analysis {outputName!r}: "
                    f"{len(opened)} matrices and {len(files)} file artifacts.\n"
                    "Current MRI images are unchanged. Product commitment occurs only "
                    "when the owning action succeeds."
                )
                if omittedRoles:
                    pglMessages.message(f"Model matrices intentionally not retained: {omittedRoles}")

                return product

            finally:
                for matrix in temporaryMatrices:
                    matrix.close()
                if not attachmentAttempted:
                    for matrix in opened.values():
                        matrix.close()
        def _afterSuccess(self, session):
            """Display committed results through the shared saved-product interface."""
            if not self.showFigures or not self.actionHistory: return

            records = [
                record for record in self.actionHistory[-1].actionProducts
                if record.get("kind") == "GLMsingleAnalysis"
            ]
            if len(records) != 1:
                pglMessages.warning("Could not identify this execution's GLMsingle analysis product.", level=1)
                return

            product = session.getProduct(productID=records[0]["productID"])
            if product is None: return
            product.display(cacheDirectory=self.cacheDirectory)

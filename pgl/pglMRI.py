################################################################
#   filename: pglMRI.py
#    purpose: Class for holding mri data used by pglSession
#         by: JLG
#       date: Oct 31, 2026
################################################################

#############
# Import modules
#############
from copy import deepcopy
from pathlib import Path

from traitlets import Any, Dict, Instance, List, Unicode

from .pglPipeline import pglActionable
from .pglSettings import pglTraitSettings


################################################################
# MRI scan
################################################################
class pglMRIScan(pglTraitSettings):
    """A local NIfTI reference with source metadata.

    Opening image reads the NIfTI header and creates a lazy data proxy.
    It does not load the complete voxel array.

    Local files must remain available and unchanged while readers use them.
    Database persistence and verified remote backing are not implemented yet.
    """

    name = Unicode("", help="Display name of scan")

    # Local paths are runtime access information, not portable checkpoint
    # locations. Database persistence will handle payload references explicitly.
    localPath = Unicode("", help="Local NIfTI filename", visible=False).tag(
        serialize=False
    )

    sourceMetadata = Dict(
        default_value={},
        help="Source identifiers and acquisition metadata",
    )

    # NiBabel objects are runtime readers, not JSON metadata.
    _image = Any(default_value=None, allow_none=True).tag(serialize=False)

    @property
    def image(self):
        """Open the image on first access without loading the voxel array."""
        if self._image is not None:
            return self._image

        try:
            import nibabel as nib
        except ImportError:
            raise ImportError(
                "MRI image access requires nibabel. "
                "Install using: pip install nibabel"
            ) from None

        if not self.localPath:
            raise ValueError("This MRI scan has no local source file.")

        path = Path(self.localPath)
        if not path.is_file():
            raise FileNotFoundError(f"MRI source file is unavailable: {path}")

        # NiBabel reads header information and retains an ArrayProxy.
        # Do not call get_fdata() here: that would load the voxel array.
        image = nib.load(str(path), mmap=True)

        if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
            raise TypeError("MRI scans currently support single-file NIfTI images.")

        # Cache only after successful opening and type validation.
        self._image = image
        return image


################################################################
# MRI collection
################################################################
class pglMRI(pglActionable):
    """Ordered MRI scans, independent of behavioral runs.

    This initial implementation attaches local NIfTI references.
    It does not copy files, publish checkpoints, or infer behavioral matching.
    """

    scans = List(
        Instance(pglMRIScan),
        help="Ordered MRI scans",
    )

    def add(self, filename, *, name=None, sourceMetadata=None):
        """Attach a local NIfTI without opening its header or voxel array.

        Returns the new pglMRIScan.

        Source metadata is copied so subsequent caller edits do not modify
        the attached record. Metadata may contain identifying subject details.

        File existence and extension are checked here. NIfTI format validation
        occurs on first image access. Recorded source hashes are not verified
        by this attachment operation.
        """
        path = Path(filename).expanduser().absolute()

        if not path.is_file():
            raise FileNotFoundError(f"MRI source file is unavailable: {path}")

        if not path.name.lower().endswith((".nii", ".nii.gz")):
            raise ValueError("MRI scans currently require .nii or .nii.gz files.")

        if name is None:
            name = path.name

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Scan name must be a nonblank string.")

        if sourceMetadata is None:
            sourceMetadata = {}

        if not isinstance(sourceMetadata, dict):
            raise TypeError("sourceMetadata must be a dictionary or None.")

        scan = pglMRIScan(
            name=name,
            localPath=str(path),
            sourceMetadata=deepcopy(sourceMetadata),
        )

        # Assign a new list so trait observers see the attachment.
        self.scans = [*self.scans, scan]
        return scan
################################################################
#   filename: pglFlywheel
#    purpose: Code for downloading data from flywheel
#         by: JLG
#       date: Aug 20, 2026
################################################################

#############å
# Import modules
#############
from .pglMessages import pglMessages
from traitlets import Float, TraitError, TraitError, observe, Instance, List, Int, Unicode, Dict, validate, Bool
from .pglSettings import pglTraitSettings

class pglFlywheelScan(pglTraitSettings):
    name = Unicode(help="Name / type of scan", visible=False)

class pglFlywheelSession(pglTraitSettings):
    name = Unicode(help='Name of session',visible=False)
    subjectID = Unicode(help='subjectID', enable=False)
    id = Unicode(help='session ID', visible=False)
    scans = List(Instance(pglFlywheelScan), help="Scans in session", settingsListKey="name", multiSelect=True, maxRowsVisible=12)

class pglFlywheelProject(pglTraitSettings):
    name = Unicode(help='Name of project',visible=False)
    id = Unicode(help='Identifier of project', visible=False)
    sessions = List(Instance(pglFlywheelSession), help='Sessions in project', settingsListKey='name')

class pglFlywheelChoose(pglTraitSettings):
    projects = List(Instance(pglFlywheelProject), settingsListKey="name", help='flywheel sessions to load data from')
        
class pglFlywheel():
    def __init__(self, apikey):
        '''
        init an instance of a connection flywheel
        '''
        # load library
        try:
            import flywheel
        except Exception as e:
            pglMessages.warning(f"unable to import flywheel library\ninstall using: pip install flywheel-sdk: {e}")
            return
        
        try:
            # initaite flywheel connection
            fw = flywheel.Client(apikey)
        except Exception as e:
            pglMessages.warning(f"Could not connect to flywheel: {e}")
            return
        
        try:
            # check user to see if we are actualy connected
            user = fw.get_current_user()
            pglMessages.message(f"Logged in as: {user.firstname} {user.lastname} ({user.email})")
        except Exception as e:
            pglMessages.warning("Login to flywheel failed:", e)
            return
        
        # walk the tree
        self.choose = pglFlywheelChoose()
        
        # walk projects
        projects = fw.projects()
        for flywheelProject in projects:
            # get project info
            project = pglFlywheelProject(name=flywheelProject.label, id=flywheelProject.id)
            # get all session under project
            flywheelSessions = fw.projects.find_first(f'label="{project.name}"').sessions()
            for flywheelSession in flywheelSessions:
                # get session info
                session = pglFlywheelSession(name=flywheelSession.timestamp.strftime('%Y-%m-%d %H:%M'), subjectID=flywheelSession.subject.label, id=flywheelSession.id)
                # get all scans
                flywheelScans = flywheelSession.acquisitions()
                for flywheelScan in flywheelScans:
                    for flywheelFile in flywheelScan.files:
                        if flywheelFile.type == "nifti":
                            scan = pglFlywheelScan(name=f"{flywheelScan.label}: {flywheelFile.name}")
                    session.scans.append(scan)
                # add session to project
                project.sessions.append(session)
            # add to choose
            self.choose.projects.append(project)

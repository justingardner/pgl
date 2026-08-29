################################################################
#   filename: pglDigitalBrain
#    purpose: Code for digital brain
#         by: JLG
#       date: Aug 23, 2026
################################################################

# import
from .pglSettings import pglTraitSettings
from traitlets import Unicode, Int, List, Tuple
from .pglExperiment import pglTask
from .pglKeyboardMouse import pglKeyBuffer
from .pglImage import pglMovieDatabase
from .pglParameter import pglParameter
from .pglMessages import pglMessages
from .pglTasks import pglMessageAckTask, pglEyeTrackingCalibrationTask
from pathlib import Path
import numpy as np

###########################################
# choose settings for block
###########################################
class pglChooseBlock(pglTraitSettings):
    subjectID = Unicode(default_value="s000", help="Subject ID")
    descriptionLength = Int(12, help="description length in seconds", min=0)
    displayWidth = Int(50, help="Size of movie width in deg", min=0)

    def __init__(self, subjectMax=5, dayMax=2, blockMax=5, **kwargs):
        super().__init__(**kwargs)

        self.add_traits(
            subjectNum=Int(
                min=0,
                max=subjectMax,
                default_value=0,
                help="Subject number for memory pilot"
            ),
            dayNum=Int(
                min=0,
                max=dayMax,
                default_value=0,
                help="Day of experiment"
            ),
            blockNum=Int(
                min=0,
                max=blockMax,
                default_value=0,
                help="Which block"
            ),
        )

def pglDigitalBrainConfigure(e, currentRun):
    if currentRun is None:
        pglMessages.warning("Must set current run parameters for each block before initializing experiment")

    # get pgl
    pgl = e.pgl

    # other settings
    descriptionLength = currentRun.descriptionLength
    displayWidth = currentRun.displayWidth

    # First run a calibration
    messageAckTask = pglMessageAckTask(pgl, "Press space to start eye calibration")
    messageAckTask.settings.phaseNum=0
    e.addTask(messageAckTask)
    calibrationTask = pglEyeTrackingCalibrationTask(pgl)
    calibrationTask.settings.phaseNum=1
    e.addTask(calibrationTask)

    # description task
    descriptionTask = pglDigitalBrainMemoryTask(pgl, subjectNum=currentRun.subjectNum, dayNum=currentRun.dayNum, blockNum=currentRun.blockNum, descriptionLength=descriptionLength, displayWidth=displayWidth)
    descriptionTask.settings.phaseNum = 2
    e.addTask(descriptionTask)

    # calibration
    messageAckTask = pglMessageAckTask(pgl, "Press any response key to start eye calibration")
    messageAckTask.settings.phaseNum = 3
    e.addTask(messageAckTask)
    calibrationTask = pglEyeTrackingCalibrationTask(pgl)
    calibrationTask.settings.phaseNum = 4
    e.addTask(calibrationTask)

    return e

###########################################
# Memory task
###########################################
class pglDigitalBrainMemoryTask(pglTask):
    
    ########################
    def __init__(self, pgl, subjectNum, dayNum, blockNum, descriptionLength=12, displayWidth=30):
        super().__init__(pgl)
        
        # initialize the key buffer
        self.keyBuffer = pglKeyBuffer(maxLineLength=40)
        
        # set task parameters, these will automatically be saved in the settings file
        self.settings.taskName = "Memory description task"
        
        # set seglens
        self.settings.seglen = [0.5, float('inf'), descriptionLength, 0.5]

        # fixed parameters, these will automatically be saved in the settings file
        self.settings.fixedParameters = {
            #'moviePath':'/Users/Shared/digital-assets/stimulus/digital/0008',
            #'moviePath':'/Users/justin/Desktop/testvideos',
            'moviePath':'/Users/Shared/digital',
            #'moviePath':'/Users/justin/Desktop/digital',
            'displayWidth': displayWidth,
            'subjectNum': subjectNum,
            'dayNum': dayNum,
            'blockNum': blockNum,
        }        
        p = self.settings.fixedParameters

        # response mappings for proper display
        self.data.responseMapping = {
            # Correct
            0: ("high confidence new", "#0072B2"),  # dark blue
            1: ("low confidence new",  "#8FC4E8"),  # light blue
            2: ("low confidence old",  "#F6BE73"),  # light orange
            3: ("high confidence old", "#D55E00"),  # dark orange-red
            4: ("remembered",          "#7B3294"),  # purple

            # Incorrect: same response category, muted / grayish version
            5: ("incorrect: high confidence new", "#6F8794"),
            6: ("incorrect: low confidence new",  "#B7C4CC"),
            7: ("incorrect: low confidence old",  "#C9B69C"),
            8: ("incorrect: high confidence old", "#9A7768"),
            9: ("incorrect: remembered",          "#8F7C96"),
        }
        
        # create the blockPath
        self.state.blockPath = Path(p['moviePath']) / f"1{subjectNum}{dayNum}{blockNum}"

        # load movie database
        self.mdb = pglMovieDatabase(self.state.blockPath)
        self.mdb.useManifest(filenameColumn="filename", indexColumn="trial_index", conditionColumn="condition")
        
        # get number of trials to run for 
        self.settings.nTrials=self.mdb.nStimuli
        
        # add parameters
        self.addParameter(pglParameter('movieNum',np.arange(self.mdb.nStimuli),randomize=False))
        self.addParameter(pglParameter('description',["description"]))
        
        # print
        print("=!"*40)
        print(f"{self.state.blockPath.name} subject={p['subjectNum']} day={p['dayNum']} block={p['blockNum']}")
        print("=!"*40)
        
        # print out the trials
        for iStimulus in range(self.mdb.nStimuli):
            m = self.mdb.stimuli[iStimulus]
            print(f"{iStimulus}: filename: {m.filename} condition: {m.condition}")
                    
                    
    ########################
    def startSegment(self, startTime):
        '''
        Start a segment.
        '''
        super().startSegment(startTime)
        
        if self.state.currentSegment == 0:
            self.e.flush = True
            # do not eat keys
            self.e.setEatAllKeys(False)
            # load the movie
            moviePath = self.mdb.stimuli[self.currentParams['movieNum']].filename
            condition = self.mdb.stimuli[self.currentParams['movieNum']].condition
            self.m = self.pgl.movie(filename=str(moviePath),displayWidth=self.settings.fixedParameters['displayWidth'])
            pglMessages.message(f"{self.state.currentTrial}: {self.m} moviePath: {moviePath} condition: {condition}")
        
        elif self.state.currentSegment == 1:
            self.state.gotResponse = False
            # play the movie
            self.m.play(displayWidth=self.settings.fixedParameters['displayWidth'])
            self.jumpSegment()

        elif self.state.currentSegment == 2:
            # description segment
            self.e.setEatAllKeys(True)
            self.keyBuffer.clear()
            self.state.keyBufferDirty=False
            self.state.elapsedTime = -1
            self.e.flush = False

        elif self.state.currentSegment == 3:
            self.e.flush = True
            self.e.setEatAllKeys(False)
            # save the description
            self.currentParams['description'] = self.keyBuffer.getText()
    ########################
    def updateScreen(self):
        if self.state.currentSegment == 0:
            pass
        elif self.state.currentSegment == 2:
            # calcluate elapsed time
            elapsedTime = round(self.settings.seglen[self.state.currentSegment]-(self.pgl.getSecs()-self.state.segmentStartTime),0)
            
            # decide if we need to draw (only if elapsed time has changed or keyBufferDirty)
            if elapsedTime != self.state.elapsedTime or self.state.keyBufferDirty:
                # update elapsed time
                self.state.elapsedTime = elapsedTime
                # draw text
                self.pgl.text(f"Describe: {elapsedTime:0.1f}", line="center")
            
                # draw the subject text
                text = self.keyBuffer.getWrappedText()
                for line in text.split('\n'):
                    self.pgl.text(line)
                self.state.keyBufferDirty = False
                
                # flush screen
                self.e.pgl.flush()
    
    ########################
    def handleEvents(self, events):
        for event in events:
            if event.eventType == 'keydown':
                self.keyBuffer.processEvent(event)
                self.state.keyBufferDirty = True

    ########################
    # handleSubjectResponse
    ########################    
    def handleSubjectResponse(self, response, updateTime):
        '''
        Handle the subject response. Returns the value 0-4 if correct see responseMapping above for explanation
        For incorrect answers returns 5-9
        '''
        
        # already received a response
        if self.state.gotResponse: return None
        # mark that we got a response
        self.state.gotResponse = True
        
        condition = self.mdb.stimuli[self.currentParams['movieNum']].condition
        if condition.lower().startswith("new"):
            # if not a new response then make it incorrect
            if response not in {0,1}: response += 5
        else:
            # if a new response then make it incorrect
            if response in {0,1}: response += 5
               
        # check if response is correct 
        return response
    
    ########################
    # handleSubjectResponse
    ########################    
    def end(self):
        '''
        end of task
        '''
        super().end()
        # make sure flush is set back to normal
        self.e.flush = True
        # do not eat keys
        self.e.setEatAllKeys(False)
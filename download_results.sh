
#!/bin/bash
# syncs model results only

# rsync -avz lyt@openmind7.mit.edu:/om/user/lyt/ec/jobs .
# rsync -avz lyt@openmind7.mit.edu:/om/user/lyt/ec/experimentOutputs .

#rsync -avz lyt@openmind7.mit.edu:/om/user/lyt/ec/experimentOutputs/draw/2019-11-10T22:47:09.086994 experimentOutputs/draw/
rsync -avz lyt@openmind7.mit.edu:/om/user/lyt/ec/experimentOutputs/draw/2020-05-1[0-9]* experimentOutputs/draw/
rsync -avz lyt@openmind7.mit.edu:/om/user/lyt/ec/experimentOutputs/draw/2020-05-2[0-9]* experimentOutputs/draw/


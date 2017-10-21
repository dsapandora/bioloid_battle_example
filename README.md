# bioloid_battle_example
BIOLOID BATTLET TRAINNING using ROS

To control de robot we are using a RBG camer ASUS based in
https://github.com/spencer-project/spencer_people_tracking

To train the dataset we use keras

loss : categorical_crossentropy
optimized: adam
adam=optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
in 1000 epochs

the inputs are 81 elements from the based in spencer tracking person  pose and tracking person twist
The ouput is a value that represent an element in the robot battle action list
bioloid_action = ["WFWD","WBWD","WLT ","WRT ","WLSD","WRSD","WFLS","WFRS","WBLS","WBRS","WAL ","WAR ","WFLT","WFRT","WBLT","WBRT","WRDY","SIT ","STND","ATKL","ATKR","ATKF", "ATKD"]
 
To learn using categorical_crossentropy we convert the ouput to matrix using to_categorical, to because this index are for real categorial
To transform back this valus to index we are using numpy argmax


The robot use a custom firware based in the project
https://github.com/dsapandora/bioloid_project_structure



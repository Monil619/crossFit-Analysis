import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns

# read csv file
df = pd.read_csv('new.csv')

# input of player names
print(" Enter name of players to be compared :")
p1 = input("Player A:")
p2 = input("Player B:")


# extract result of player
r1 = df[df['Name'] == p1]
r2 = df[df['Name'] == p2]

plot = pd.DataFrame({
'Player': [p1,p2],
'Aerobic': [r1.iloc[0]['18.1_percentile'], r2.iloc[0]['18.1_percentile']],
'Anaerobic': [r1.iloc[0]['18.2_percentile'], r2.iloc[0]['18.2_percentile']],
'Gymnastic': [r1.iloc[0]['18.3_percentile'], r2.iloc[0]['18.3_percentile']],
'Weightlifting': [r1.iloc[0]['18.4_percentile'], r2.iloc[0]['18.4_percentile']],
'strength': [r1.iloc[0]['18.5_percentile'], r2.iloc[0]['18.5_percentile']]
})

#18.1 workout- Aerobic workout
#18.2 workout- Anaerobic workout
#18.3 workout- Gymnastic workout
#18.4 workout- Weightlifting workout
#18.5 workout- strength workout

#label
labels=np.array(['Aerobic', 'Anaerobic', 'Gymnastic', 'Weightlifting', 'strength'])

#set angle
angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)

#close angle
angles=np.concatenate((angles,[angles[0]]))

#start plot
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw xlabels
#plt.xticks(angles[:-1], categories)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([80,85,90,95], ["80","85","90","95"], color="grey", size=8)
plt.ylim(75,100)

#1st player plot
stats=plot.loc[0,labels].values
stats=np.concatenate((stats,[stats[0]]))

ax.plot(angles, stats, linewidth=1, linestyle = 'solid', label = p1)
ax.fill(angles, stats,'b', alpha=0.2)
ax.set_thetagrids(angles * 180/np.pi, labels)

#2nd player plot
stats=plot.loc[1,labels].values
stats=np.concatenate((stats,[stats[0]]))


ax.plot(angles, stats, linewidth=1, linestyle = 'solid', label = p2)
ax.fill(angles, stats,'r', alpha=0.2)
ax.set_thetagrids(angles * 180/np.pi, labels)

#plot legend
plt.legend(loc='upper right', bbox_to_anchor=(1.45,1.1))

#save plot in file
plt.savefig(p1+' vs. '+p2+'.png')

ax.grid(True)

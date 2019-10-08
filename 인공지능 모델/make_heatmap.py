import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

vegetables = ['official', 'musician', 'comedian', 'actor', 'soccer',
          'basketball',  'Baseball', 'athlete', 'poet', 'soldier',
          'businessmen', 'scholar', 'player', 'moviedirector', 'progamer',
          'religionist', 'journalist', 'artist', 'author']

farmers = ['official', 'musician', 'comedian', 'actor', 'soccer',
          'basketball',  'Baseball', 'athlete', 'poet', 'soldier',
          'businessmen', 'scholar', 'player', 'moviedirector', 'progamer',
          'religionist', 'journalist', 'artist', 'author']

harvest = np.array([[19, 0, 0, 0,   0,   0,   1,   0,   0,   2,   0,   1,   0,   0,   0,   0,   1,   0,   0],
[0, 10, 1, 2,   0,   0 ,  0 ,  1 ,  0 ,  1 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  1 ,  0 ,  0],
[0, 0, 11, 1,   0,   0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  1 ,  0 ,  1],
[0, 3, 4, 13,   0,   0 ,  0 ,  1 ,  0 ,  0 ,  1 ,  1 ,  0 ,  1 ,  0 ,  0 ,  1 ,  0 ,  1],
[0, 0, 0, 0,  20 ,  2  , 1  , 2  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0],
[0, 0, 0, 0,   1 , 17  , 0  , 2  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0],
[0, 0, 0, 0,   0 ,  0  ,20  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0],
[1, 0, 0, 0,   0 ,  1  , 0  ,16  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0],
[1, 2, 0, 0,   0 ,  0  , 0  , 0  ,11  , 5  , 1  , 2  , 2  , 0  , 0  , 0  , 0  , 0  , 0],
[1, 0, 0, 1,   0 ,  0  , 0  , 0  , 0  ,18  , 1  , 1  , 2  , 0  , 0  , 0  , 0  , 0  , 0],
[0, 0, 1, 0,   1 ,  0  , 0  , 0  , 0  , 5  ,17  , 0  , 0  , 0  , 0  , 1  , 1  , 0  , 0],
[1, 1, 0, 0,   1 ,  0  , 1  , 0  , 1  , 1  , 0  ,10  , 0  , 0  , 1  , 0  , 4  , 0  , 1],
[0, 0, 0, 1,   0 ,  0  , 0  , 0  , 0  , 1  , 0  , 0  , 9  , 2  , 0  , 0  , 0  , 0  , 0],
[0, 0, 0, 0,   0 ,  0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 6  , 1  , 0  , 2  , 0  , 1],
[1, 0, 1, 0,   0 ,  0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  ,20  , 0  , 0  , 1  , 0],
[3, 1, 2, 0,   0 ,  0  , 0  , 1  , 3  , 1  , 0  , 2  , 2  , 1  , 0  , 8  , 0  , 0  , 0],
[1, 1, 0, 0,   0 ,  0  , 0  , 2  , 0  , 0  , 3  , 1  , 0  , 0  , 0  , 0  , 9  , 0  , 0],
[0, 0, 0, 0,   0 ,  0  , 0  , 1  , 0  , 2  , 1  , 3  , 0  , 2  , 0  , 0  , 0  , 6  , 1],
[0, 1, 0, 0, 0,   0,    0,   0   ,0   ,0   ,0   ,3   ,0   ,0   ,0   ,0   ,1   ,0   ,9]])


fig, ax = plt.subplots()
im = ax.imshow(harvest, cmap="BuPu")

# We want to show all ticks...
ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))
# ... and label them with the respective list entries
ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

ax.set_title("RNN Job Classification Result")



fig.tight_layout()

plt.show()
plt.savefig('heat_map.jpg')

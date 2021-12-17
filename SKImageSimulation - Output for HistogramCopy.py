from IPython.core.display import display, HTML
import numpy as np
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2
from collections import Counter
import pg_fitter_tools as fit
import sk_geo_tools as sk
import os
import csv
import math



####################Specify File Inputs###############################################
filename = "045.jpg"
textFilename = "045-manualLabelling.txt"
# Output file will contain these initials next to coordinates:
initials = "MS"
WIDTH = 4000
HEIGHT = 2750
######################################################################################

####################Settings##########################################################
plotReprojectedPMTs = True
plotReprojectedFeatures = False
plotManualFeatures = True
inputFileType = "manualLabelling"
#inputFileType = "imageProcessing"

offset = 0
suppressInput = False
labellingStats = True
######################################################################################

####################Internal Camera Parameters########################################

focal_length = [2.760529621789217e+03, 2.767014510543478e+03]
principle_point = [1.914303537872458e+03, 1.596386868474348e+03]
radial_distortion = [-0.2398, 0.1145]
tangential_distortion = [0, 0]

######################################################################################

####################Specify External Camera Parameters################################
#Image 045.JPG:
#rotation_vector = np.array([[1.52593994],[-0.71901074],[0.60290209]])
#translation_vector = np.array([[-100.74973094],[1606.91543897],[-916.79105257]])
######################################################################################

######################################################################################
####################Specify External Camera Parameters################################
#Image 045.JPG:

rotation_vector = np.array([[-np.pi/2],[0.0],[0.0]])
translation_vector = np.array([[0.0],[0.0],[-750]])

##left/right, vertical, back/front
#translation_vector = np.array([[500],[500],[-800.0]])
######################################################################################




####################Plot points#######################################################
def plot_pmts(coordinates, imageIdentifier, off_set=0, color=(0,255,0)):
    counter = 0
    for i in coordinates:
        if np.abs(int(i[0]))<=4000 and np.abs(int(i[1])-int(off_set))<=2750:
            plotx = int(i[0])
            ploty = int(i[1])-int(off_set)
            cv2.circle(imageIdentifier,(plotx,ploty),7,color,-1)
            counter=counter+1
######################################################################################

####################Obtain Reprojected Points#########################################
def obtain_reprojected_points(features):
    nfeatures = len(features)

    seed_feature_locations = np.zeros((nfeatures, 3))
    feature_index = {}
    index_feature = {}
    f_index = 0
    for f_key, f in features.items():
        feature_index[f_key] = f_index
        index_feature[f_index] = f_key
        seed_feature_locations[f_index] = f
        f_index += 1

    
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    transformed_positions = (rotation_matrix @ seed_feature_locations.T).T + translation_vector.T
    indices = np.where(transformed_positions[:,2]>0)[0]

    camera_matrix = build_camera_matrix(focal_length, principle_point)
    distortion = build_distortion_array(radial_distortion, tangential_distortion)

    reprojected = cv2.projectPoints(seed_feature_locations[indices], rotation_vector, translation_vector,camera_matrix, distortion)[0].reshape((indices.size, 2))

## Can't do the filtering here.

    
    reprojected_points = {}
    reprojected_points[filename_no_extension] = dict(zip([index_feature[ii] for ii in indices], reprojected))



#    reprojected_filtered = []
#    for i in reprojected:
#        if i[0]<=WIDTH and i[0]>=0 and i[1]<=HEIGHT and i[1]>=0:
#            reprojected_filtered.append([i[0],i[1]])#
#
#    reprojected = reprojected_filtered

    for point in list(reprojected_points[filename_no_extension]):
        #print(reprojected_points[filename_no_extension][point][0])
        if(reprojected_points[filename_no_extension][point][0]>WIDTH or reprojected_points[filename_no_extension][point][0]<0 or reprojected_points[filename_no_extension][point][1]>HEIGHT or reprojected_points[filename_no_extension][point][1]<0):
            reprojected_points[filename_no_extension].pop(point)


    return reprojected_points
######################################################################################


def build_camera_matrix(focal_length, principle_point):
    return np.array([
        [focal_length[0], 0, principle_point[0]],
        [0, focal_length[1], principle_point[1]],
        [0, 0, 1]], dtype=float)


def build_distortion_array(radial_distortion, tangential_distortion):
    return np.concatenate((radial_distortion, tangential_distortion)).reshape((4, 1))


def plotter(points, color, marker, label):
    plotLegend = False
    for i in points:
        #print(i[0],i[1],i[2])
        if(plotLegend == False):
            ax.plot([float(i[0])],[float(i[1])],[float(i[2])],color = color, marker = marker, label=label)
            plotLegend = True
        else:
            ax.plot([float(i[0])],[float(i[1])],[float(i[2])],color = color, marker = marker)
    
def plotter_labels(points):
    for i in points:
        ax.text(float(i[0]),float(i[1]),float(i[2])+10, i[3][-3:],size=12, zorder=4, color='#5158fc')


# Create output text file:
filename_no_extension = os.path.splitext(filename)[0]
camera_rotations = np.zeros((1, 3))
camera_translations = np.zeros((1, 3))
camera_rotations[0, :] = rotation_vector.ravel()
camera_translations[0, :] = translation_vector.ravel()

#outputTextFilename = os.path.join(filename_no_extension+"-"+inputFileType+" - angle to vertical - centre of G.txt")
#print("Creating file for writing output:",outputTextFilename)
#outputFile = open(outputTextFilename,"w")


# Read all 3D PMT locations
all_pmts = fit.read_3d_feature_locations("parameters/SK_all_PMT_locations.txt")

all_bolts = sk.get_bolt_locations_barrel(all_pmts)

all_pmt_coords = np.stack(list(all_pmts.values()))




# Obtain 2D reprojected feature points
#pmt_reprojected_points, pmt_world_points = obtain_reprojected_points(all_pmts)
pmt_reprojected_points = obtain_reprojected_points(all_pmts)
pmt_repro_coords = np.stack(list(pmt_reprojected_points[filename_no_extension].values()))

pmt_world_points = []
for item in pmt_reprojected_points[filename_no_extension]:
    #print(item)
    for pmt in all_pmts:
        if(pmt == item):
            pmt_world_points.append((all_pmts[item][0],all_pmts[item][1],all_pmts[item][2]))
            #print(item, all_pmts[item][0], all_pmts[item][1],all_pmts[item][2])
pmt_world_points_array = np.array(pmt_world_points)

camera_orientations, camera_positions = fit.camera_world_poses(camera_rotations, camera_translations)
camera_position_coords = [float(camera_positions[0]), float(camera_positions[1]), float(camera_positions[2])]









###################### Reproject Using New Points:
#New camera parameters


def reprojection_simulation(name="default",
                            focal_length = [2.760529621789217e+03, 2.767014510543478e+03],
                            principle_point = [1.914303537872458e+03, 1.596386868474348e+03],
                            radial_distortion = [-0.2398, 0.1145],
                            tangential_distortion = [0, 0],
                            rotation_vector = np.array([[-np.pi/2],[0.0],[0.0]]),
                            translation_vector = np.array([[0.0],[0.0],[-750]])):

    camera_matrix = build_camera_matrix(focal_length, principle_point)
    distortion = build_distortion_array(radial_distortion, tangential_distortion)
    #print(pmt_world_points_array)
    #new_reprojected_points = cv2.projectPoints(pmt_world_points_array, rotation_vector, translation_vector, camera_matrix, distortion)[0].reshape((len(pmt_world_points_array), 2))
    new_pmt_reprojected_points = obtain_reprojected_points(all_pmts)
    new_pmt_repro_coords = np.stack(list(new_pmt_reprojected_points[filename_no_extension].values()))

    #print("3D point length is ", len(pmt_world_points_array))
    #print("Original 2D projected point lenght is ", len(pmt_repro_coords))
    #print("2D projected point length is ", len(new_pmt_reprojected_points))

    x,y = zip(*pmt_repro_coords)
    #print(x,y)
#    plt.plot(x, y, color='green', marker='o', linewidth=0)
#    x2,y2 = zip(*new_pmt_repro_coords)
    #print(x2, y2)
#    plt.plot(x2,y2, color = 'red', marker='o', linewidth=0)
#    plt.ylim(HEIGHT, 0)
#    plt.savefig(name + "_new_reprojected_points.png")
#    plt.show()

#    plt.close()

    points_x = []
    points_y = []
    errors_x = []
    errors_y = []
    distances = []
    for point in pmt_reprojected_points[filename_no_extension]:
        
        for new_point in new_pmt_reprojected_points[filename_no_extension]:
            if(new_point == point):
                #print("point and new_point: ", point, new_point, "new x: ", new_pmt_reprojected_points[filename_no_extension][new_point][1], "old x", pmt_reprojected_points[filename_no_extension][point][1])
                
                distance_x = pmt_reprojected_points[filename_no_extension][point][0]-new_pmt_reprojected_points[filename_no_extension][new_point][0]
                distance_y = pmt_reprojected_points[filename_no_extension][point][1]-new_pmt_reprojected_points[filename_no_extension][new_point][1]
                points_x.append((pmt_reprojected_points[filename_no_extension][point][0] + new_pmt_reprojected_points[filename_no_extension][new_point][0])/2)
                points_y.append((pmt_reprojected_points[filename_no_extension][point][1] + new_pmt_reprojected_points[filename_no_extension][new_point][1])/2)
                #print(distance_x)
                #print(distance_y)
                
                errors_x.append(abs(distance_x))
                errors_y.append(abs(distance_y))
                distances.append(np.sqrt(distance_x**2 + distance_y**2))

    return points_x,points_y,errors_x,errors_y,distances
    #print(errors_x, errors_y)

def plot_errors(name, errors_x,errors_y):    
    errorAverage_x = sum(errors_x) / len(errors_x)
    average_errors_x.append(errorAverage_x)
    errorAverage_y = sum(errors_y) / len(errors_y)
    average_errors_y.append(errorAverage_y)


#    print("Errors in x and y = \n", errorAverage_x,"\n",errorAverage_y)
    fig = plt.figure(figsize=(12,12))
    n, bins, patches = plt.hist(errors_x,20, facecolor='g')
    plt.xlabel('Error (pixels)', size=25)
    plt.ylabel('Frequency', size=25)
    title = 'Distance Errors from changing ' + name + ", X axis"
    plt.title(title, size=30, wrap=True)


    STD_x = np.std(errors_x)
    STD_x_error = STD_x / np.sqrt(len(errors_x))
    printSTD = "SD = " + str(round(STD_x,1)) + " cm"
    printAverage = "AVG = " + str(round(errorAverage_x,2)) + " ± " + str(round(STD_x_error,2)) + " cm"


    plt.text(0.007, 10, printAverage,fontsize=30)
    plt.text(0.007, 6, printSTD, fontsize = 30)
    plt.tick_params(axis='both', labelsize=20)
    #plt.xlim(0, .25)
    #plt.ylim(0, 5)
    plt.grid(True)
    plt.savefig(name + "_error-x.png")
#    plt.show()
    plt.close()
    return(errorAverage_x, errorAverage_y)



parameter = "radial distortion 1"
study = 0.0005
average_errors_x = []
average_errors_y = []

name = parameter+" by "+str(study)+" mm"
focal_length = [2760.529621789217, 2767.014510543478]
principle_point = [1914.303537872458, 1596.386868474348]
radial_distortion = [-0.2398+study, 0.1145]
tangential_distortion = [0, 0]
camera_matrix = build_camera_matrix(focal_length, principle_point)
distortion = build_distortion_array(radial_distortion, tangential_distortion)
rotation_vector = np.array([[-np.pi/2],[0.0],[0.0]])
translation_vector = np.array([[0.0],[0.0],[-750]])
points_x,points_y,errors_x,errors_y,errors = reprojection_simulation(name, radial_distortion )

file = open(name+".txt", "w")
file.write("#x\ty\terrors\n")
for i in range(len(points_x)):
    #print(points_x[i], "\t",points_y[i], "\t", errors[i])
    file.write("%f\t%f\t%f\n" %(points_x[i],points_y[i],errors[i]))
#errorAverage_x, errorAverage_y = plot_errors(name,errors_x,errors_y)
file.close()


parameter = "radial distortion 2"
study = 0.0005
average_errors_x = []
average_errors_y = []

name = parameter+" by "+str(study)+" mm"
focal_length = [2760.529621789217, 2767.014510543478]
principle_point = [1914.303537872458, 1596.386868474348]
radial_distortion = [-0.2398, 0.1145+study]
tangential_distortion = [0, 0]
camera_matrix = build_camera_matrix(focal_length, principle_point)
distortion = build_distortion_array(radial_distortion, tangential_distortion)
rotation_vector = np.array([[-np.pi/2],[0.0],[0.0]])
translation_vector = np.array([[0.0],[0.0],[-750]])
points_x,points_y,errors_x,errors_y,errors = reprojection_simulation(name, radial_distortion )

file = open(name+".txt", "w")
file.write("#x\ty\terrors\n")
for i in range(len(points_x)):
    #print(points_x[i], "\t",points_y[i], "\t", errors[i])
    file.write("%f\t%f\t%f\n" %(points_x[i],points_y[i],errors[i]))
#errorAverage_x, errorAverage_y = plot_errors(name,errors_x,errors_y)
file.close()


parameter = "focal length x"
study = 1
average_errors_x = []
average_errors_y = []

name = parameter+" by "+str(study)+" mm"
focal_length = [2760.529621789217+study, 2767.014510543478]
principle_point = [1914.303537872458, 1596.386868474348]
radial_distortion = [-0.2398, 0.1145]
tangential_distortion = [0, 0]
camera_matrix = build_camera_matrix(focal_length, principle_point)
distortion = build_distortion_array(radial_distortion, tangential_distortion)
rotation_vector = np.array([[-np.pi/2],[0.0],[0.0]])
translation_vector = np.array([[0.0],[0.0],[-750]])
points_x,points_y,errors_x,errors_y,errors = reprojection_simulation(name, radial_distortion )

file = open(name+".txt", "w")
file.write("#x\ty\terrors\n")
for i in range(len(points_x)):
    #print(points_x[i], "\t",points_y[i], "\t", errors[i])
    file.write("%f\t%f\t%f\n" %(points_x[i],points_y[i],errors[i]))
#errorAverage_x, errorAverage_y = plot_errors(name,errors_x,errors_y)
file.close()

parameter = "focal length y"
study = 1
average_errors_x = []
average_errors_y = []

name = parameter+" by "+str(study)+" mm"
focal_length = [2760.529621789217, 2767.014510543478+study]
principle_point = [1914.303537872458, 1596.386868474348]
radial_distortion = [-0.2398, 0.1145]
tangential_distortion = [0, 0]
camera_matrix = build_camera_matrix(focal_length, principle_point)
distortion = build_distortion_array(radial_distortion, tangential_distortion)
rotation_vector = np.array([[-np.pi/2],[0.0],[0.0]])
translation_vector = np.array([[0.0],[0.0],[-750]])
points_x,points_y,errors_x,errors_y,errors = reprojection_simulation(name, radial_distortion )

file = open(name+".txt", "w")
file.write("#x\ty\terrors\n")
for i in range(len(points_x)):
    #print(points_x[i], "\t",points_y[i], "\t", errors[i])
    file.write("%f\t%f\t%f\n" %(points_x[i],points_y[i],errors[i]))
#errorAverage_x, errorAverage_y = plot_errors(name,errors_x,errors_y)
file.close()


parameter = "principle point x"
study = 1
average_errors_x = []
average_errors_y = []

name = parameter+" by "+str(study)+" mm"
focal_length = [2760.529621789217, 2767.014510543478]
principle_point = [1914.303537872458+study, 1596.386868474348]
radial_distortion = [-0.2398, 0.1145]
tangential_distortion = [0, 0]
camera_matrix = build_camera_matrix(focal_length, principle_point)
distortion = build_distortion_array(radial_distortion, tangential_distortion)
rotation_vector = np.array([[-np.pi/2],[0.0],[0.0]])
translation_vector = np.array([[0.0],[0.0],[-750]])
points_x,points_y,errors_x,errors_y,errors = reprojection_simulation(name, radial_distortion )

file = open(name+".txt", "w")
file.write("#x\ty\terrors\n")
for i in range(len(points_x)):
    #print(points_x[i], "\t",points_y[i], "\t", errors[i])
    file.write("%f\t%f\t%f\n" %(points_x[i],points_y[i],errors[i]))
#errorAverage_x, errorAverage_y = plot_errors(name,errors_x,errors_y)
file.close()


parameter = "principle point y"
study = 1
average_errors_x = []
average_errors_y = []

name = parameter+" by "+str(study)+" mm"
focal_length = [2760.529621789217, 2767.014510543478]
principle_point = [1914.303537872458, 1596.386868474348+study]
radial_distortion = [-0.2398, 0.1145]
tangential_distortion = [0, 0]
camera_matrix = build_camera_matrix(focal_length, principle_point)
distortion = build_distortion_array(radial_distortion, tangential_distortion)
rotation_vector = np.array([[-np.pi/2],[0.0],[0.0]])
translation_vector = np.array([[0.0],[0.0],[-750]])
points_x,points_y,errors_x,errors_y,errors = reprojection_simulation(name, radial_distortion )

file = open(name+".txt", "w")
file.write("#x\ty\terrors\n")
for i in range(len(points_x)):
    #print(points_x[i], "\t",points_y[i], "\t", errors[i])
    file.write("%f\t%f\t%f\n" %(points_x[i],points_y[i],errors[i]))
#errorAverage_x, errorAverage_y = plot_errors(name,errors_x,errors_y)
file.close()






'''
cv2.namedWindow(filename,cv2.WINDOW_NORMAL)
cv2.moveWindow(filename, 500, 0)
img = cv2.imread(filename)

plot_pmts(pmt_repro_coords, img)
for i in pmt_repro_coords:
    text = str(round(i[0])) + str(' ') + str(round(i[1]))
    cv2.putText(img, f'{text}', (int(i[0]),int(i[1])), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 1)

cv2.imwrite("OUTPUT.jpg",img)
cv2.imshow(filename,img)
'''








      

















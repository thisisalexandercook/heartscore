import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import os
import re
import imageio



def generateData(HRV_lower_bound, HRV_upper_bound, WHR_lower_bound, WHR_upper_bound,mean, range_, w_mean, w_range_):

    cluster_size = int(100)
    std_dev = range_ / 4
    w_std_dev = w_range_ / 4



    # ABSOLUTE UNIFORM SIMULATION METHOD
    # HRV_lower_bound = 20
    # HRV_upper_bound = 100
    # WHR_lower_bound = 40
    # WHR_upper_bound = 90
    # HRV_data1 = np.random.uniform(HRV_lower_bound+(0.5*(HRV_upper_bound - HRV_lower_bound)), HRV_upper_bound, size=(cluster_size, 1))
    # WHR_data1 = np.random.uniform(WHR_lower_bound, WHR_upper_bound-(0.75*(WHR_upper_bound - WHR_lower_bound)), size=(cluster_size, 1))
    # HRV_data2 = np.random.uniform(HRV_lower_bound+(0.25*(HRV_upper_bound - HRV_lower_bound)), HRV_upper_bound-(0.25*(HRV_upper_bound - HRV_lower_bound)), size=(cluster_size, 1))
    # WHR_data2 = np.random.uniform(WHR_upper_bound-(0.75*(WHR_upper_bound - WHR_lower_bound)), WHR_upper_bound-(0.5*(WHR_upper_bound - WHR_lower_bound)), size=(cluster_size, 1))
    # HRV_data3 = np.random.uniform(HRV_lower_bound, HRV_upper_bound-(0.5*(HRV_upper_bound - HRV_lower_bound)), size=(cluster_size, 1))
    # WHR_data3 = np.random.uniform(WHR_lower_bound+(0.5*(WHR_upper_bound - WHR_lower_bound)), WHR_upper_bound, size=(cluster_size, 1))

    # NORMALLY DISTRIBUTED SIMULATION METHOD

    HRV_data1 = np.random.normal(mean+15, std_dev, size=(cluster_size, 1))
    HRV_data2 = np.random.normal(mean+10, std_dev, size=(cluster_size, 1))
    HRV_data3 = np.random.normal(mean, std_dev, size=(cluster_size, 1))

    WHR_data1 = np.random.normal(w_mean-10, w_std_dev, size=(cluster_size, 1))
    WHR_data2 = np.random.normal(w_mean-5, w_std_dev, size=(cluster_size, 1))
    WHR_data3 = np.random.normal(w_mean, w_std_dev, size=(cluster_size, 1))


    xA = np.concatenate((HRV_data1, WHR_data1), axis=1)
    xB = np.concatenate((HRV_data2, WHR_data2), axis=1)
    xC = np.concatenate((HRV_data3, WHR_data3), axis=1)

    xMain = np.concatenate((xA, xB, xC), axis=0)
    dataLabel = np.empty(len(xMain), dtype = int)
    dataLabel.fill(9)

    bounds = [(HRV_lower_bound, HRV_upper_bound), (WHR_lower_bound, WHR_upper_bound)]


    centroid_grabber = ScreeSim(HRV_lower_bound, HRV_upper_bound, WHR_lower_bound, WHR_upper_bound, xMain)
    print("CENTROID GRABBER")
    print(centroid_grabber)

    # c1 = np.array([np.random.randint(low, high) for low, high in bounds])
    c1 = centroid_grabber[0]
    c1 = c1.reshape(1,2)
    # c2 = np.array([np.random.randint(low, high) for low, high in bounds])
    c2 = centroid_grabber[1]
    c2 = c2.reshape(1,2)
    # c3 = np.array([np.random.randint(low, high) for low, high in bounds])
    c3 = centroid_grabber[2]
    c3 = c3.reshape(1,2)
    cMain = np.concatenate((c1, c2, c3), axis=0)

    return xMain, cMain, xA, xB, xC, c1, c2, c3, dataLabel,


def plotData(c1, c2, c3, xMain, dataLabel,plot_title):
    # Create an array with the same shape as dataLabel containing the row indices
    index_array = np.arange(dataLabel.shape[0])

    # Update the mask to include the condition on the row index
    mask1 = (dataLabel == 1) & (index_array > 299)
    mask2 = (dataLabel == 2) & (index_array > 299)
    mask3 = (dataLabel == 3) & (index_array > 299)

    plt.plot(xMain[dataLabel==9 , 0], xMain[dataLabel==9, 1], 'ko', alpha=0.25)
    plt.plot(xMain[dataLabel==1, 0], xMain[dataLabel==1, 1], 'bo', alpha=0.25)
    plt.plot(xMain[dataLabel==2, 0], xMain[dataLabel==2, 1], 'ro', alpha=0.25)
    plt.plot(xMain[dataLabel==3, 0], xMain[dataLabel==3, 1], 'go', alpha=0.25)
    plt.plot(xMain[mask1, 0], xMain[mask1, 1], 'D', color=(0, 0.2, 1), markersize = 10, markeredgecolor='k', markeredgewidth=2, alpha=0.5)
    plt.plot(xMain[mask2, 0], xMain[mask2, 1], 'D', color=(1, 0, 0.5), markersize = 10, markeredgecolor='k', markeredgewidth=2, alpha=0.5)
    plt.plot(xMain[mask3, 0], xMain[mask3, 1], 'D', color=(0, 1, 0.2), markersize = 10, markeredgecolor='k', markeredgewidth=2, alpha=0.5)
    plt.plot(c1[:, 0], c1[:, 1], 'bo', markersize=15)
    plt.plot(c2[:, 0], c2[:, 1], 'ro', markersize=15)
    plt.plot(c3[:, 0], c3[:, 1], 'go', markersize=15)
    plt.xlabel('Heart Rate Variability (ms)')
    plt.ylabel('Waking Heart Rate (bpm)')
    plt.title(plot_title)
    plt.axis('square')
    plt.figure()
    plt.show()

def InitialplotData(c1, c2, c3, xMain, dataLabel,plot_title):

    plt.plot(xMain[dataLabel==9, 0], xMain[dataLabel==9, 1], 'ko', alpha=0.25)
    plt.plot(xMain[dataLabel==1, 0], xMain[dataLabel==1, 1], 'bo', alpha=0.25)
    plt.plot(xMain[dataLabel==2, 0], xMain[dataLabel==2, 1], 'ro', alpha=0.25)
    plt.plot(xMain[dataLabel==3, 0], xMain[dataLabel==3, 1], 'go', alpha=0.25)
    plt.xlabel('Heart Rate Variability (ms)')
    plt.ylabel('Waking Heart Rate (bpm)')
    plt.title(plot_title)
    plt.axis('square')
    plt.figure()
    plt.show()




def centroidDistance(x, c):

    dataDistance = np.array(math.sqrt((x[0, 0] - c[0, 0]) ** 2 + (x[0, 1] - c[0, 1]) ** 2))


    for i in range(1, len(x)):
        dataDistance = np.append(dataDistance, (math.sqrt((x[i, 0] - c[0, 0]) ** 2 + (x[i, 1] - c[0, 1]) ** 2)))

    return dataDistance


def labelData(dist1, dist2, dist3):

    dataLabel = np.array(min(dist1[0], dist2[0], dist3[0]))

    if dataLabel == dist1[0]:
        dataLabel = 1
    elif dataLabel == dist2[0]:
        dataLabel = 2
    elif dataLabel == dist3[0]:
        dataLabel = 3

    for i in range(1, len(dist1)):

        dataLabel = np.append(dataLabel, min(dist1[i], dist2[i], dist3[i]))

        if dataLabel[i] == dist1[i]:
            dataLabel[i] = 1
        elif dataLabel[i] == dist2[i]:
            dataLabel[i] = 2
        elif dataLabel[i] == dist3[i]:
            dataLabel[i] = 3

    return dataLabel


def centroidMean(xMain, dataLabel,c1,c2,c3):

    totalC1 = 0
    totalC2 = 0
    totalC3 = 0
    c1x = 0
    c2x = 0
    c3x = 0
    c1y = 0
    c2y = 0
    c3y = 0

    for i in range((len(xMain))):

        if dataLabel[i] == 1:
            totalC1 += 1
            c1x += xMain[i, 0]
            c1y += xMain[i, 1]
        elif dataLabel[i] == 2:
            totalC2 += 1
            c2x += xMain[i, 0]
            c2y += xMain[i, 1]
        elif dataLabel[i] == 3:
            totalC3 += 1
            c3x += xMain[i, 0]
            c3y += xMain[i, 1]

    if totalC1 > 0:
        newC1 = np.array([[c1x / totalC1, c1y / totalC1]])
    else:
        newC1 = c1
    if totalC2 > 0:
        newC2 = np.array([[c2x / totalC2, c2y / totalC2]])
    else:
        newC2 = c2
    if totalC3 >0:
        newC3 = np.array([[c3x / totalC3, c3y / totalC3]])
    else:
        newC3 = c3

    return newC1, newC2, newC3

def rank(c1,c2,c3):
    rc1 = c1
    rc2 = c2
    rc3 = c3

    cHRV_list = [c1[0,0], c2[0,0], c3[0,0]]
    cHRV_max = max(cHRV_list)
    cHRV_min = min(cHRV_list)

    if c1[0,0] == cHRV_max:
        rc1 = c1
    elif c2[0,0] == cHRV_max:
        rc1 = c2
    else:
        rc1 = c3

    if c1[0,0] == cHRV_min:
        rc3 = c1
    elif c2[0,0] == cHRV_min:
        rc3 = c2
    else:
        rc3 = c3

    if c1[0,0] != cHRV_min and c1[0,0] != cHRV_max:
        rc2 = c1
    elif c2[0,0] != cHRV_min and c2[0,0] != cHRV_max:
        rc2 = c2
    else:
        rc2 = c3

    return rc1, rc2, rc3

def sortBySlope(c1, c2, c3):
    cArray = np.concatenate((c1, c2, c3), axis=0) #create 3x2 array with centroids
    slpsAndCs = np.c_[np.ones(3), cArray]  #add a column of 1's to the RH side
    for i in range(3):
        slpsAndCs[i, 0] = cArray[i, 1] / cArray[i, 0]  #put the slope for each c to RH side of each row

    sortedSlps = np.ones((3, 4))
    slpsAndCs = np.c_[np.array([1,2,3]), slpsAndCs]

    maxSlope = max(slpsAndCs[0, 1], slpsAndCs[1, 1], slpsAndCs[2, 1])
    minSlope = min(slpsAndCs[0, 1], slpsAndCs[1, 1], slpsAndCs[2, 1])

    #sort the rows from slpsAndCs by slopes, from max to min
    for i in range(3):
        if slpsAndCs[i,1] == maxSlope:
            sortedSlps[0] = slpsAndCs[i]
        elif slpsAndCs[i,1] == minSlope:
            sortedSlps[2] = slpsAndCs[i]
        else:
            sortedSlps[1] = slpsAndCs[i]
    return sortedSlps

def ScreeSim(HRV_lower_bound, HRV_upper_bound, WHR_lower_bound, WHR_upper_bound, xMain):

    bounds = [(HRV_lower_bound, HRV_upper_bound), (WHR_lower_bound, WHR_upper_bound)]

    iterations = 10
    k_range = 10
    k_intra_values = np.zeros(k_range)
    centroid_holder = np.zeros((iterations, 3, 2))

    for k in range(1,k_range + 1):

        intra_arr = np.zeros(iterations)
        for counter in range(iterations):

            centroids = np.zeros((k, 2))
            centroids_difference = np.zeros(len(centroids))
            centroids_difference.fill(10)
            for i in range(k):
                centroids[i] = np.array([np.random.randint(low, high) for low, high in bounds])

            if k == 3:
                centroid_holder[counter] = centroids

            previous_centroids = centroids
            while np.any(centroids_difference > 0.1):

                distance = np.zeros((k, len(xMain)))
                for x in range(k):
                    distance[x] = ScreeDistance(centroids[x], xMain)
                data_labels = np.argmin(distance, axis=0)
                centroids = ScreeUpdateCentroid(xMain, centroids, data_labels)

                centroids_difference = np.linalg.norm(previous_centroids - centroids)

            intra_arr[counter] = ScreeIntra(xMain, centroids, data_labels)

        k_intra_values[k-1] = np.min(intra_arr)
        k_intra_index = np.argmin(intra_arr)

    print("AVERAGE INTRA-CLUSTER VALUES FOR K = 1-10")
    print(k_intra_values)
    ScreePlot(k_intra_values, k_range)

    return centroid_holder[k_intra_index]


def ScreePlot(k_intra_values, cluster_count):

    plt.plot(np.arange(1, cluster_count+1), k_intra_values, color='black')
    plt.scatter(np.arange(1, cluster_count+1), k_intra_values, color='black')
    plt.xticks(np.arange(1, 11))

    plt.title('Athlete Scree Plot')
    plt.xlabel('Cluster Count - K')
    plt.ylabel('Objective value - Intra Distance')

    plt.show()


def ScreeUpdateCentroid(xMain, centroids, data_labels):

    new_centroids = centroids

    for x in range(len(centroids)):
        selected_points = xMain[data_labels == x]
        new_centroids[x] = np.mean(selected_points, axis=0)

    return new_centroids


def ScreeIntra(xMain, centroids, data_labels):

    avg_distance = 0

    for x in range(len(xMain)):
        avg_distance += np.linalg.norm(xMain[x] - centroids[data_labels[x]])

    avg_distance /= len(xMain)

    return avg_distance


def ScreeDistance(centroid, xMain):

    distance = np.zeros(len(xMain))

    for x in range(len(xMain)):
        distance[x] = np.linalg.norm(xMain[x] - centroid)

    return distance


def RunSimulation(HRV_lower_bound, HRV_upper_bound, WHR_lower_bound, WHR_upper_bound, plot_title, mean, range_, w_mean, w_range_):

    xMain, cMain, xA, xB, xC, c1, c2, c3, dataLabel, = generateData(HRV_lower_bound, HRV_upper_bound, WHR_lower_bound, WHR_upper_bound, mean, range_, w_mean, w_range_)
    InitialplotData(c1, c2, c3, xMain, dataLabel,plot_title)
    plotData(c1, c2, c3, xMain, dataLabel,plot_title)
    centroidDifferenceC1 = 100
    centroidDifferenceC2 = 100
    centroidDifferenceC3 = 100

    while centroidDifferenceC1 > 0.1 and centroidDifferenceC2 > 0.1 and centroidDifferenceC3 > 0.1:

        previousC1 = c1
        previousC2 = c2
        previousC3 = c3

        dist1 = centroidDistance(xMain, c1)
        dist2 = centroidDistance(xMain, c2)
        dist3 = centroidDistance(xMain, c3)
        dataLabel = labelData(dist1, dist2, dist3)
        c1, c2, c3 = centroidMean(xMain, dataLabel, c1, c2, c3)
        plotData(c1, c2, c3, xMain, dataLabel, plot_title)

        centroidDifferenceC1 = np.array(math.sqrt((previousC1[0, 0] - c1[0, 0]) ** 2 + (previousC1[0, 1] - c1[0, 1]) ** 2))
        centroidDifferenceC2 = np.array(math.sqrt((previousC2[0, 0] - c2[0, 0]) ** 2 + (previousC2[0, 1] - c2[0, 1]) ** 2))
        centroidDifferenceC3 = np.array(math.sqrt((previousC3[0, 0] - c3[0, 0]) ** 2 + (previousC3[0, 1] - c3[0, 1]) ** 2))

    print("")
    print("Cluster Rankings by slope")
    print(sortBySlope(c1, c2, c3))

    return c1, c2, c3, dataLabel, xMain

def UpdateSimulation(c1, c2, c3, xMain, dataLabel,plot_title):

    InitialplotData(c1, c2, c3, xMain, dataLabel,plot_title)
    plotData(c1, c2, c3, xMain, dataLabel,plot_title)
    centroidDifferenceC1 = 100
    centroidDifferenceC2 = 100
    centroidDifferenceC3 = 100

    while centroidDifferenceC1 > 0.1 and centroidDifferenceC2 > 0.1 and centroidDifferenceC3 > 0.1:

        previousC1 = c1
        previousC2 = c2
        previousC3 = c3

        dist1 = centroidDistance(xMain, c1)
        dist2 = centroidDistance(xMain, c2)
        dist3 = centroidDistance(xMain, c3)
        dataLabel = labelData(dist1, dist2, dist3)
        c1, c2, c3 = centroidMean(xMain, dataLabel, c1, c2, c3)
        plotData(c1, c2, c3, xMain, dataLabel, plot_title)

        centroidDifferenceC1 = np.array(math.sqrt((previousC1[0, 0] - c1[0, 0]) ** 2 + (previousC1[0, 1] - c1[0, 1]) ** 2))
        centroidDifferenceC2 = np.array(math.sqrt((previousC2[0, 0] - c2[0, 0]) ** 2 + (previousC2[0, 1] - c2[0, 1]) ** 2))
        centroidDifferenceC3 = np.array(math.sqrt((previousC3[0, 0] - c3[0, 0]) ** 2 + (previousC3[0, 1] - c3[0, 1]) ** 2))

    print("")
    print("Cluster Rankings by slope")
    print(sortBySlope(c1, c2, c3))

    return c1, c2, c3, dataLabel, xMain


def UserInterface():

    print("Welcome to Heartscore")

    while True:
        print("1) Athlete Simulation")
        print("2) Non-Athlete Simulation")
        print("3) Exit")
        user_select = input("Please select a simulation type (1) or (2) or (3) to exit: ")

        if user_select == "1":
            print("Running athlete simulation")
            plot_title = "Athlete Simulation"
            c1, c2, c3, dataLabel, xMain = RunSimulation(37, 71, 52, 65, plot_title,54, 30, 57, 10)
            break
        elif user_select == "2":
            print("Running Non-Athlete simulation")
            plot_title = "Non-Athlete Simulation"
            c1, c2, c3, dataLabel, xMain = RunSimulation(24, 62, 63, 84, plot_title,43, 19, 72, 13)
            break
        elif user_select == "3":
            print("Goodbye")
            break
        else:
            print("Please make an appropriate selection")

    if user_select == "1" or user_select == "2":
        AddData(c1, c2, c3, dataLabel, xMain,plot_title)

def AddData(c1, c2, c3, dataLabel, xMain,plot_title):
    update_counter = 0
    while True:

        if update_counter > 0:
            data_question = input("Would you like to add additional datapoints to classify OR update your model? (y/n/u)")
        else:
            data_question = input("Would you like to add additional datapoints to classify? (y/n)")

        if data_question == "y":
            HRV_input = float(input("Please enter HRV in ms: "))
            WHR_input = float(input("Please enter WHR in bpm: "))
            new_point = np.array([[HRV_input, WHR_input]])
            xMain = np.concatenate((xMain, new_point), axis=0)
            dist1 = centroidDistance(xMain, c1)
            dist2 = centroidDistance(xMain, c2)
            dist3 = centroidDistance(xMain, c3)
            new_data_label = np.empty(len(xMain), dtype=int)
            new_data_label.fill(9)
            new_data_label = labelData(dist1,dist2,dist3)
            plotData(c1, c2, c3, xMain, new_data_label, plot_title)
            slope_rank = sortBySlope(c1,c2,c3)
            if new_data_label[-1] == slope_rank[2, 0]:
                print("You're well rested, training hard today is advised")
            elif new_data_label[-1] == slope_rank[1, 0]:
                print("You're partially rested, moderate traning is advised")
            elif new_data_label[-1] == slope_rank[0, 0]:
                print("You're not well rested, taking a rest day is advised")
            update_counter += 1

        elif data_question == "n":
            print("Goodbye")
            break

        elif data_question == "u":
            print("updating model")
            xMain = xMain[update_counter:]
            dataLabel = np.empty(len(xMain), dtype=int)
            dataLabel.fill(9)
            c1, c2, c3, dataLabel, xMain = UpdateSimulation(c1, c2, c3, xMain, dataLabel,plot_title)
            update_counter = 0
        else:
            print("Please enter a valid response")


def IntraCluster(c1, c2, c3, dataLabel, xMain):

    if dataLabel[0] == 1:
        dataDistance = np.array(math.sqrt((xMain[0, 0] - c1[0, 0]) ** 2 + (xMain[0, 1] - c1[0, 1]) ** 2))
    elif dataLabel[0] == 2:
        dataDistance = np.array(math.sqrt((xMain[0, 0] - c2[0, 0]) ** 2 + (xMain[0, 1] - c2[0, 1]) ** 2))
    elif dataLabel[0] == 3:
        dataDistance = np.array(math.sqrt((xMain[0, 0] - c3[0, 0]) ** 2 + (xMain[0, 1] - c3[0, 1]) ** 2))



## DIAGRAM UTILS BELOW

def PhotoGrid(rows,columns,dir,size_x,size_y):

    path = dir

    images = [i for i in os.listdir(path) if i.endswith('.png')]
    images.sort(key=sort_key)
    print(images)

    grid_size = (rows, columns)  # Changed to 4x3

    img_list = [Image.open(os.path.join(path, i)).resize((size_x, size_y)) for i in images]  # Change the size as needed
    grid_img = Image.new('RGB', (size_x * grid_size[0], size_y * grid_size[1]))

    for i in range(grid_size[1]):
        for j in range(grid_size[0]):
            grid_img.paste(img_list[i * grid_size[0] + j], (j * size_x, i * size_y))

    grid_img.save(os.path.join(path, "grid_image.png"))

def sort_key(s):
    return int(re.match(r'A(\d+).png', s).group(1))


def DistanceDiagram():
    data_points = np.array([[40, 52], [59, 53], [63, 63], [62, 49]])
    colors = ['blue', 'green', 'red', 'black']
    labels = ['centroid 1', 'centroid 2', 'centroid 3', 'data point']
    alphas = [1, 1, 1, 0.25]
    sizes = [150, 150, 150, 100]
    offset = 0.5

    for i in range(4):
        plt.scatter(data_points[i, 0], data_points[i, 1], c=colors[i], s=sizes[i], alpha=alphas[i])
        plt.text(data_points[i, 0] + offset, data_points[i, 1], labels[i])

    for i in range(3):
        plt.plot([data_points[i, 0], data_points[3, 0]], [data_points[i, 1], data_points[3, 1]], color=colors[i])

    for i in range(3):
        distance = np.linalg.norm(data_points[3] - data_points[i])
        plt.text((data_points[i, 0] + data_points[3, 0]) / 2,
                 (data_points[i, 1] + data_points[3, 1]) / 2,
                 f'{distance:.2f}',
                 fontsize=9,
                 ha='right')

    plt.xlabel('Heart Rate Variability (ms)')
    plt.ylabel('Waking Heart Rate (bpm)')
    plt.title("K-means distance measurement")
    plt.axis('square')
    plt.show()

def CreateGif(dir):


    input_dir = dir
    output_gif = "output.gif"
    files = os.listdir(input_dir)

    png_files = [i for i in files if i.endswith('.png')]

    png_files.sort(key=sort_key)

    with imageio.get_writer(output_gif, mode='I') as writer:
        for filename in png_files:
            image = imageio.imread(os.path.join(input_dir, filename))
            writer.append_data(image)

    print("GIF created successfully!")


## CLIENT CODE

UserInterface()
# PhotoGrid(3,4,"AthleteGrid",640,480)
# DistanceDiagram()
# CreateGif("AthleteGrid")


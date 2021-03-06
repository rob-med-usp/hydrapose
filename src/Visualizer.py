import time
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import sys

from .SkeletonsBridge import SkeletonsBridge

class Visualizer:
    
    def __init__(self):
        self.bridge = SkeletonsBridge()
        self.skeleton_color = 'dodgerblue'
        self.camma_colors_skeleton = ['y', 'g', 'g', 'g', 'g', 'm', 'm', 'm', 'm', 'm']
    
    def init3D(self, total, pos, ground = False):
        # 3D plot axis
        self.ax3D = self.fig.add_subplot(1, total, pos, projection = '3d')
        # Set right visualization position 
        self.ax3D.azim = -90
        self.ax3D.dist = 10
        self.ax3D.elev = -60
        # space around the persons
        RADIUS = 1500 
        self.ax3D.set_xlim([-RADIUS, RADIUS])
        self.ax3D.set_ylim([-RADIUS ,RADIUS])
        self.ax3D.set_zlim([0, 2*RADIUS])
        
        self.ax3D.set_xlabel('x')
        self.ax3D.set_ylabel('y')
        self.ax3D.set_zlabel('z')
        

        if ground:
            self.onlyGroundAx(self.ax3D)

    def initWindowsWithDepth(self):
        
        # Init figure
        self.fig = plt.figure()
        # Image axis
        self.axImage = self.fig.add_subplot(1,3,1)
        # Depth image axis
        self.axDepth = self.fig.add_subplot(1,3,2)
        # init 3D plot
        self.init3D(3,3)
    
    def initWindows(self):
        # Init figure
        self.fig = plt.figure()
        # Image axis
        self.axImage = self.fig.add_subplot(1,2,1)
        # init 3D plot
        self.init3D(2,2, ground=False)

    def drawSkeleton(self, image, persons2D, mode = 'Human36M', upper_body = True):
        
        if mode == 'Human36M':
            if upper_body == True:
                pairs = self.bridge.pairs_upper_Human36M_noseless
            else:
                pairs = self.bridge.pairs_Human36M_noseless
        elif mode == 'MPII':
            if upper_body == True:
                pairs = self.bridge.pairs_upper_MPII
            else:
                pairs = self.bridge.pairs_MPII
        elif mode == 'COCO':
            if upper_body == True:
                pairs = self.bridge.pairs_upper_COCO
            else:
                pairs = self.bridge.pairs_COCO
        elif mode == 'MVOR':
            pairs = self.bridge.pairs_MVOR
        
        # Iterate for each person
        for person in persons2D:
            # iterate for each pair
            for pair in pairs:
                part1 = pair[0]
                part2 = pair[1]
                point1 = tuple(person[part1].astype(int))
                point2 = tuple(person[part2].astype(int))
                # Draw lines
                # color = (np.random.randint(255),np.random.randint(255),np.random.randint(255))
                color = (255,165,0)
                cv2.line(image, point1, point2, color, 3)
    
    def drawPersonAxis(self, image, point_zero, points_axis):
        img = cv2.line(image, point_zero, tuple(np.array(points_axis[0].ravel(), np.uint)), (255,0,0), 5)
        img = cv2.line(image, point_zero, tuple(np.array(points_axis[1].ravel(), np.uint)), (0,255,0), 5)
        img = cv2.line(image, point_zero, tuple(np.array(points_axis[2].ravel(), np.uint)), (0,0,255), 5)
        return img

    def onlyGroundAx(self, ax3D):
        
        # Get rid of the ticks and tick labels
        ax3D.set_xticks([])
        ax3D.set_yticks([])
        ax3D.set_zticks([])

        ax3D.get_xaxis().set_ticklabels([])
        ax3D.get_yaxis().set_ticklabels([])
        ax3D.set_zticklabels([])

        # Get rid of the panes (actually, make them white)
        white = (1.0, 1.0, 1.0, 1.0)
        ax3D.w_xaxis.set_pane_color(white)
        ax3D.w_zaxis.set_pane_color(white)
        # Keep z pane

        # Get rid of the lines in 3d
        ax3D.w_xaxis.line.set_color(white)
        ax3D.w_yaxis.line.set_color(white)
        ax3D.w_zaxis.line.set_color(white)
    
    def drawBones3D(self, ax3D, keypoints3D, pairs, color = []):
        
        # x, y, z = [], [], []
        # if upper_body:
        #     for keypoint in range(len(keypoints3D)):
        #         if(keypoints3D[keypoint][0] != -1) and (keypoint in self.pos_upperbody):
        #             x.append(keypoints3D[keypoint][0])
        #             y.append(keypoints3D[keypoint][1])
        #             z.append(keypoints3D[keypoint][2])
        # else:
        #     for keypoint in range(len(keypoints3D)):
        #         if(keypoints3D[keypoint][0] != -1):
        #             x.append(keypoints3D[keypoint][0])
        #             y.append(keypoints3D[keypoint][1])
        #             z.append(keypoints3D[keypoint][2])
            
        for idx, pair in enumerate(pairs):
                
            if(-1 in keypoints3D[pair[0]])  or (-1 in keypoints3D[pair[1]]):
                continue
            
            x1 = keypoints3D[pair[0]][0]
            y1 = keypoints3D[pair[0]][1]
            z1 = keypoints3D[pair[0]][2]

            x2 = keypoints3D[pair[1]][0]
            y2 = keypoints3D[pair[1]][1]
            z2 = keypoints3D[pair[1]][2]
            
            # ax3D.plot([-x1, -x2],[-z1, -z2] ,[-y1, -y2])
            if not color:
                ax3D.plot([x1, x2],[y1, y2], [z1, z2], color = 'red')
            
            elif color:
                ax3D.plot([x1, x2],[y1, y2], [z1, z2], color = color[idx])

        # # Draw keypoints
        # x = np.array(x)
        # y = np.array(y)
        # z = np.array(z)
        # # ax3D.scatter(-x, -z, -y)
        # ax3D.scatter(x, y, z)
        
        return ax3D
    
    def show(self, image, persons3D, depth=[], block = False, mode = 'Human36M'):
        
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = image
        self.axImage.imshow(image_rgb)

        if hasattr(self, 'axDepth'):
            self.axDepth.imshow(depth)
        # Check if list is not empty
        elif depth:
            # Call error
            print("You must initialize depth image plot using hy.initWindowsWithDepth()")
        
        color = []
        paris = []
        if mode == 'Human36M':
            #TODO
            upper_body = True

            if upper_body is True:
                pairs = self.bridge.pairs_upper_Human36M_noseless
            else:
                pairs = self.bridge.pairs_Human36M_noseless

        elif mode == 'MVOR':
            pairs = self.bridge.pairs_MVOR
            color = self.camma_colors_skeleton
        # Plot keypoints 3D
        # i = 0
        for person in persons3D:
            # if i>=len(persons3D)/2:
            #     self.skeleton_color="magenta"
            # else:
            #     self.skeleton_color="orange"
            # i+=1
            if not color:
                self.ax3D = self.drawBones3D(self.ax3D, person, pairs)
            elif color:
                self.ax3D = self.drawBones3D(self.ax3D, person, pairs, color=color)
        
        # Display 3D plot
        plt.show(block=block)
        
        if block is False:
            plt.pause(1)

        if block is True:
            plt.pause(-1)
        
    def comparePlot3D(self, persons1, persons2):

        fig = plt.figure()
        ax3D = fig.add_subplot(1, 1, 1, projection = '3d')
        # Set right visualization position 
        ax3D.azim = -90
        ax3D.dist = 10
        ax3D.elev = -60
        # space around the persons
        RADIUS = 1500 
        ax3D.set_xlim([-RADIUS, RADIUS])
        ax3D.set_ylim([-RADIUS ,RADIUS])
        ax3D.set_zlim([0, 2*RADIUS])

        ax3D.set_xlabel('x')
        ax3D.set_ylabel('y')
        ax3D.set_zlabel('z')
        viz = Visualizer()
        for person in persons1[:,:10]:
            ax3D = viz.drawBones3D(ax3D, person, self.bridge.pairs_MVOR, color=viz.camma_colors_skeleton)
        for person in persons2[:,:10]:
            ax3D = viz.drawBones3D(ax3D, person, self.bridge.pairs_MVOR)
        plt.show()
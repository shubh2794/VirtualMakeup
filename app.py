# import the necessary packages
from tkinter import *
from tkinter.ttk import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2
from applications import makeup
from skimage import io
import dlib
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

class Checkbar(Frame):
	def __init__(self, parent=None, picks=[], side=LEFT, anchor=W):
		Frame.__init__(self, parent)
		self.vars = []
		for pick in picks:
			var = IntVar()
			chk = Checkbutton(self, text=pick, variable=var)
			chk.pack(side=side, anchor=anchor, expand=YES)
			self.vars.append(var)
	def state(self):
		return map((lambda var: var.get()), self.vars)

class Makeup:
	def __init__(self, window, window_title):

		self.window = window
		self.window.title(window_title)

		self.top = Frame(self.window)
		self.bottom = Frame(self.window)
		self.top.pack(side=TOP)
		self.bottom.pack(side=BOTTOM, fill=BOTH, expand=True)

		self.colors_df = pd.read_csv('colors.csv', delimiter='\t')
		self.colors = self.colors_df['ColorName'].tolist()
		# print(colors_df[colors_df['ColorName'] == 'orange']['Code'])

		self.panelA = None
		self.panelB = None
		self.opencv_image = None

		self.lipstick_count = 0
		self.kajal_count = 0
		self.blush_count = 0

		self.btn1 = Button(self.window, text="Apply Makeup", command=self.apply_makeup)
		self.btn1.pack(in_=self.bottom, side='bottom', fill="both", expand="yes", padx="10", pady="10")
		
		self.lipstick_colour_options = self.colors
		self.lipstick_colour_variable = StringVar(self.window)
		self.lipstick_colour_variable.set('Select Lipstick Colour') # default value
		self.c1 = Combobox(self.window, textvariable = self.lipstick_colour_variable, values=self.lipstick_colour_options, state='readonly')
		self.c1.pack(in_=self.bottom, side="left", fill="both", expand="yes", padx="10", pady="10")
		self.c1.current(0)
		self.c1.bind("<<ComboboxSelected>>", lambda f1: self.fun())
		self.lipstick_colour = self.c1.get()

		self.eyeshadow_colour_options = self.colors
		self.eyeshadow_colour_variable = StringVar(self.window)
		self.eyeshadow_colour_variable.set('Select Eye Shadow Colour') # default value
		self.c2 = Combobox(self.window, textvariable=self.eyeshadow_colour_variable, values=self.eyeshadow_colour_options, state='readonly')
		self.c2.pack(in_=self.bottom, side="left", fill="both", expand="yes", padx="10", pady="10")
		self.c2.current(0)
		self.c2.bind("<<ComboboxSelected>>", lambda f2: self.fun2())
		self.eyeshadow_colour = self.c2.get()

		self.selectMakeup = Checkbar(self.window, ['Lipstick', 'Foundation', 'Blush', 'Eyebrow', 'Liner', 'Eyeshadow'])
		self.selectMakeup.pack(side=BOTTOM,  fill=X)

		self.btn = Button(self.window, text="Select an image", command=self.select_image)
		self.btn.pack(side='bottom', fill="both", expand="yes", padx="10", pady="10")

		self.window.mainloop()

	def fun(self):
		self.lipstick_colour = self.c1.get()

    
	def fun2(self):
		self.eyeshadow_colour = self.c2.get()

	def select_image(self):
		# grab a reference to the image panels
		global panelA , panelB
		# open a file chooser dialog and allow the user to select an input
		# image
		self.path = filedialog.askopenfilename()
		# ensure a file path was selected
		if len(self.path) > 0:
			self.image = cv2.resize(cv2.imread(self.path), (320, 640))
			self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
			self.img = self.image.copy()
			self.image = io.imread(self.path)
			self.detector = dlib.get_frontal_face_detector()
			self.face_pose_predictor = dlib.shape_predictor("face_landmarks.dat")
			self.detected_faces = self.detector(self.image, 0)
			self.pose_landmark = self.face_pose_predictor(self.image, self.detected_faces[0])
			self.landmark = np.empty([68, 2], dtype=int)

			for i in range(68):
				self.landmark[i][0] = self.pose_landmark.part(i).x
				self.landmark[i][1] = self.pose_landmark.part(i).y
                
			self.m = makeup(self.image)
            

			# convert the images to PIL format...
			self.img = Image.fromarray(self.img)
			# res = Image.fromarray(opencv_image)
			# ...and then to ImageTk format
			self.img = ImageTk.PhotoImage(self.img)
			# res = ImageTk.PhotoImage(res)
			# if the panels are None, initialize them
			if self.panelA is None or self.panelB is None:
				# the first panel will store our original image
				self.panelA = Label(image=self.img)
				self.panelA.img = self.img
				self.panelA.pack(side="left")
				# while the second panel will stre the result image
				# panelB = Label(image=res)
				# panelB.image = res
				# panelB.pack(side="right")
			# otherwise, update the image panels
			else:
				# update the pannels
				self.panelA.configure(image=self.img)
				# panelB.configure(image=res)
				self.panelA.img = self.img
				# panelB.image = res

	def apply_makeup(self):
        
		self.image = io.imread(self.path)
		self.m = makeup(self.image)
		
		self.makeupState = list(self.selectMakeup.state())

		self.lc = self.colors_df.loc[self.colors_df['ColorName'] == self.lipstick_colour, 'Code'].iloc[0].replace('(', '').replace(')', '').split(',')
		self.ec = self.colors_df.loc[self.colors_df['ColorName'] == self.eyeshadow_colour, 'Code'].iloc[0].replace('(', '').replace(')', '').split(',')

		self.lc = list(map(lambda x : int(x), self.lc))
		self.ec = list(map(lambda x : int(x), self.ec))
		
		if self.makeupState[0] == 1:
			self.im_copy = self.m.apply_lipstick(self.landmark, self.lc[0], self.lc[1], self.lc[2])
		if self.makeupState[1] == 1:
			self.im_copy = self.m.apply_foundation(self.landmark)
		if self.makeupState[2] == 1:
			self.im_copy = self.m.apply_blush(self.landmark, 223., 91., 111.)
		if self.makeupState[3] == 1:
			self.im_copy = self.m.apply_eyebrow(self.landmark)
		if self.makeupState[4] == 1:
			self.im_copy = self.m.apply_liner(self.landmark)
		if self.makeupState[5] == 1:
			self.im_copy = self.m.apply_eyeshadow(self.landmark, self.ec[0], self.ec[1], self.ec[2])
		if self.makeupState == [0,0,0,0,0,0]:
			self.im_copy = self.image.copy()

		self.im_copy = cv2.resize(self.im_copy, (320, 640))

		# convert the images to PIL format...
		self.res = Image.fromarray(self.im_copy)

		self.res = Image.fromarray(self.im_copy)

		self.res = ImageTk.PhotoImage(self.res)
		if self.panelB is None:
			self.panelB = Label(image=self.res)
			self.panelB.image = self.res
			self.panelB.pack(side="right")
		else:
			self.panelB.configure(image=self.res)
			self.panelB.image = self.res

Makeup(Tk(), "Virtual Makeup")

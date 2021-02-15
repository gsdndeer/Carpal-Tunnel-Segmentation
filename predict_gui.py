import os 
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import utils.gui as ui
from utils.gui import Ui_MainWindow
from utils.predict import *
import time

class Main(QtWidgets.QMainWindow):
     def __init__(self):
          super(Main, self).__init__()
          self.ui = Ui_MainWindow()
          self.ui.setupUi(self)               
          self.ui.pushButton.clicked.connect(self.choose_sequense)
          self.ui.horizontalSlider.valueChanged.connect(self.change_img)
          self.ui.pushButton_2.clicked.connect(self.run)


     def choose_sequense(self):
          global directory
          directory = QFileDialog.getExistingDirectory(self,"選取資料夾","./")
          DIR = directory + "/T1"
          count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]) 
          self.ui.label_3.setText("1 / %s" % count)
          
          self.ui.horizontalSlider.setMinimum(1)
          self.ui.horizontalSlider.setMaximum(count)
          self.ui.horizontalSlider.setValue(1)
          
          pixmapT1 = QPixmap (directory + "/T1/0.jpg")
          self.ui.label_4.setPixmap(pixmapT1)
          self.ui.label_4.setScaledContents (True)
               
          pixmapT2 = QPixmap (directory + "/T2/0.jpg")
          self.ui.label_5.setPixmap(pixmapT2)
          self.ui.label_5.setScaledContents (True)
               
          pixmapCT = QPixmap (directory + "/CT/0.jpg")
          self.ui.label_7.setPixmap(pixmapCT)
          self.ui.label_7.setScaledContents (True)
          
          pixmapFT = QPixmap (directory + "/FT/0.jpg")
          self.ui.label_8.setPixmap(pixmapFT)
          self.ui.label_8.setScaledContents (True)
               
          pixmapMN = QPixmap (directory + "/MN/0.jpg")
          self.ui.label_6.setPixmap(pixmapMN)
          self.ui.label_6.setScaledContents (True)


     def change_img(self):
          global val
          val = self.ui.horizontalSlider.value()
          self.ui.label_3.setText("%s" % val)
          
          pixmapT1 = QPixmap (directory + "/T1/"+str(val-1) +".jpg")
          self.ui.label_4.setPixmap(pixmapT1)
          self.ui.label_4.setScaledContents (True)
               
          pixmapT2 = QPixmap (directory + "/T2/"+str(val-1) +".jpg")
          self.ui.label_5.setPixmap(pixmapT2)
          self.ui.label_5.setScaledContents (True)
               
          pixmapCT = QPixmap (directory + "/CT/"+str(val-1) +".jpg")
          self.ui.label_7.setPixmap(pixmapCT)
          self.ui.label_7.setScaledContents (True)
          
          pixmapFT = QPixmap (directory + "/FT/"+str(val-1) +".jpg")
          self.ui.label_8.setPixmap(pixmapFT)
          self.ui.label_8.setScaledContents (True)
               
          pixmapMN = QPixmap (directory + "/MN/"+str(val-1) +".jpg")
          self.ui.label_6.setPixmap(pixmapMN)
          self.ui.label_6.setScaledContents (True)

          if  os.path.isfile("./result/CT/"+str(val-1) +".jpg"):
               CT = "./result/CT/"+str(val-1) +".jpg"
               FT = "./result/FT/"+str(val-1) +".jpg"
               MN = "./result/MN/"+str(val-1) +".jpg"

               pixmapCT = QPixmap ("./result/CT/"+str(val-1) +".jpg")
               self.ui.label_10.setPixmap(pixmapCT)
               self.ui.label_10.setScaledContents (True)

               pixmapFT = QPixmap ("./result/FT/"+str(val-1) +".jpg")
               self.ui.label_9.setPixmap(pixmapFT)
               self.ui.label_9.setScaledContents (True)
                    
               pixmapMN = QPixmap ("./result/MN/"+str(val-1) +".jpg")
               self.ui.label_11.setPixmap(pixmapMN)
               self.ui.label_11.setScaledContents (True)    

               img = Contours(directory + "/T1/"+str(val-1) +".jpg", CT, FT, MN) 
               h,w,_ = img.shape
               img = QImage(img.data, h, w, 3*h, QImage.Format_RGB888)
               self.ui.label_12.setPixmap(QPixmap(img))     
               self.ui.label_12.setScaledContents (True) 

               self.ui.label_2.setText("Current image DC :\n\nCarpal tunnel:"+str(dc_CT[val-1])+"\n\nFlexor tendons:"+str(dc_FT[val-1])+"\n\nMeidan nerve:"+str(dc_MN[val-1]))                     


     def run(self):
          start = time.time()
          global dc_CT
          global dc_FT
          global dc_MN
          global CT
          global FT
          global MN

          # load data
          print('load data...')
          img_pre = load_data(directory + "/T1")
          img_pre += load_data(directory + "/T2")
          mask_gt_CT = read_gt(load_data(directory + "/CT"))
          mask_gt_FT = read_gt(load_data(directory + "/FT"))
          mask_gt_MN = read_gt(load_data(directory + "/MN"))

          # predict
          print('predict...')
          modelpath = "./model"          
          save_img_FT = []
          save_img_CT = []
          save_img_MN = []
          for model in range(5):
               print("\rmodel {}:(CT/FT/MN)".format(model+1))
               save_img_CT.append(
                    predict(img_pre, "{}/CT/{}/model.pth".format(modelpath, model+1)))
               save_img_FT.append(
                    predict(img_pre, "{}/FT/{}/model.pth".format(modelpath, model+1)))
               save_img_MN.append(
                    predict(img_pre, "{}/MN/{}/model.pth".format(modelpath, model+1)))

          # vote
          img_num = int(len(img_pre)/2)
          pred_CT = vote(save_img_CT,  img_num)
          pred_FT = vote(save_img_FT, img_num)
          pred_MN = vote(save_img_MN, img_num)

          # save result
          save_pre_img(pred_CT, pred_FT, pred_MN)
          
          # DC
          dc_CT = DC(pred_CT, mask_gt_CT)
          dc_FT = DC(pred_FT, mask_gt_FT)
          dc_MN = DC(pred_MN, mask_gt_MN)

          # show result on GUI
          CT = "./result/CT/"+str(val-1) +".jpg"
          FT = "./result/FT/"+str(val-1) +".jpg"
          MN = "./result/MN/"+str(val-1) +".jpg"

          self.ui.label_10.setPixmap(QPixmap (CT))
          self.ui.label_10.setScaledContents (True)

          self.ui.label_9.setPixmap(QPixmap (FT))
          self.ui.label_9.setScaledContents (True)
               
          self.ui.label_11.setPixmap(QPixmap (MN))
          self.ui.label_11.setScaledContents (True)     

          img = Contours(directory + "/T1/"+str(val-1) +".jpg", CT, FT, MN) 
          h,w,_ = img.shape
          img = QImage(img.data, h, w, 3*h, QImage.Format_RGB888)
          self.ui.label_12.setPixmap(QPixmap(img))    
          self.ui.label_12.setScaledContents (True) 
          
          self.ui.label.setText("Sequence DC(mean) :\n\nCarpal tunnel:"+str(dc_CT[len(dc_CT)-1])+"\n\nFlexor tendons:"+str(dc_FT[len(dc_FT)-1])+"\n\nMeidan nerve:"+str(dc_MN[len(dc_MN)-1]))
          self.ui.label_2.setText("Current image DC :\n\nCarpal tunnel:"+str(dc_CT[val-1])+"\n\nFlexor tendons:"+str(dc_FT[val-1])+"\n\nMeidan nerve:"+str(dc_MN[val-1]))
          end = time.time()
          print(end-start)


if __name__ == '__main__':
     import sys
     app = QtWidgets.QApplication(sys.argv)
     window = Main()
     window.show()
     sys.exit(app.exec_())
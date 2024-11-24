from PIL import Image, ImageEnhance
from PIL.ImageQt import ImageQt
from palette import *
from util import *
from transfer import *
import numpy as np
from PyQt6.QtGui import QImage, QPixmap, QColor
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QComboBox, QColorDialog
from PyQt6.QtCore import Qt
from harmonization import auto_palette

# Helper functions
html_color = lambda color: '#%02x%02x%02x' % (color[0], color[1], color[2])
color_np = lambda color: np.array([color.red(), color.green(), color.blue()])

class Window(QWidget):
    K = 0 
    palette_button = []
    Source = ''
    image_label = None
    img = None
    img_lab = None
    palette_color = (np.zeros((7, 3)) + 239).astype(int)  # initial grey
    sample_level = 16
    sample_colors = sample_RGB_color(sample_level)
    sample_weight_map = []
    means = []
    means_weight = []

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Palette Based Photo Recoloring')
        self.UiComponents()
        self.show()

    def palette2mean(self):  # rgb to lab
        mean = np.zeros(self.palette_color.shape)
        for i in range(self.palette_color.shape[0]):
            rgb = Image.new('RGB', (1, 1), html_color(self.palette_color[i]))
            mean[i] = rgb2lab(rgb).getdata()[0]
        return mean.astype(int)

    def mean2palette(self):  # lab to rgb
        palette = np.zeros(self.means.shape)
        for i in range(self.means.shape[0]):
            lab = Image.new('LAB', (1, 1), html_color(self.means[i].astype(int)))
            palette[i] = lab2rgb(lab).getdata()[0]
        return palette.astype(int)

    def calc_palettes(self, k):
        self.K = k
        colors = self.img_lab.getdata()
        bins = {}
        for pixel in colors:
            bins[pixel] = bins.get(pixel, 0) + 1
        bins = sample_bins(bins)
        self.means, self.means_weight = k_means(bins, k=self.K, init_mean=True)
        print(self.means)
        self.palette_color = self.mean2palette()
        self.set_palette_color()
        print('original palette')
        print(self.palette_color)

    def pixmap_open_img(self, k):
        # load image
        self.img = Image.open(self.Source)
        print(self.Source, self.img.format, self.img.size, self.img.mode)
        # transfer to lab
        self.img_lab = rgb2lab(self.img)
        # get palettes
        self.calc_palettes(k)
        pixmap = QPixmap.fromImage(ImageQt(self.img))
        return pixmap

    def style_transfer(self):
        file_name, _ = QFileDialog.getOpenFileName(
        self,
        "Open File",
        "",
        "Images (*.jpg *.JPG *.jpeg *.png *.webp *.tiff *.tif *.bmp *.dib);;All Files (*)"
        )
        if not file_name:
            return
        # load image
        style_img = Image.open(file_name)
        # transfer to lab
        style_img_lab = rgb2lab(style_img)
        # get palettes
        colors = style_img_lab.getdata()
        bins = {}
        for pixel in colors:
            bins[pixel] = bins.get(pixel, 0) + 1
        bins = sample_bins(bins)
        style_means, _ = k_means(bins, k=self.K, init_mean=True)
        print('style', style_means)

        # Change GUI palette color
        style_palette = np.zeros(style_means.shape)
        for i in range(self.means.shape[0]):
            lab = Image.new('LAB', (1, 1), html_color(style_means[i].astype(int)))
            style_palette[i] = lab2rgb(lab).getdata()[0]
        self.palette_color = style_palette.astype(int)
        self.set_palette_color()

        # Transfer
        self.img = img_color_transfer(
            self.img_lab, self.means, style_means, 
            self.sample_weight_map, self.sample_colors, self.sample_level
        )
        print('Done')
        resized = QPixmap.fromImage(ImageQt(self.img))
        
        # resized.scaledToHeight()
        self.image_label.setPixmap(resized)

    def auto(self):
        print(self.palette_color)
        self.palette_color = auto_palette(self.palette_color, self.means_weight)
        self.set_palette_color()
        print(self.palette_color)
        # modify image
        self.img = img_color_transfer(
            self.img_lab, self.means, self.palette2mean(),
            self.sample_weight_map, self.sample_colors, self.sample_level
        )
        print('Done')
        resized = QPixmap.fromImage(ImageQt(self.img))
        
        # resized.scaledToHeight()
        self.image_label.setPixmap(resized)

    def clicked(self, N):
        if N >= self.K:
            print('invalid palette')
            return
        print('change palette', N, 'to', end='\t')
        # choose new color
        curr_clr = self.palette_color[N]
        current = QColor(curr_clr[0], curr_clr[1], curr_clr[2])
        color = QColorDialog.getColor(initial=current)
        if not color.isValid():
            return
        self.palette_color[N] = color_np(color)
        self.set_palette_color()
        # modify image
        self.img = img_color_transfer(
            self.img_lab, self.means, self.palette2mean(),
            self.sample_weight_map, self.sample_colors, self.sample_level
        )
        print('Done')
        resized = QPixmap.fromImage(ImageQt(self.img))
        
        # resized.scaledToHeight()
        self.image_label.setPixmap(resized)

    def init_palette_color(self):
        for i in range(7):
            attr = 'background-color:' + html_color(self.palette_color[i]) + ';border:0px'
            self.palette_button[i].setStyleSheet(attr)

    def set_palette_color(self):
        for i in range(self.K):
            attr = 'background-color:' + html_color(self.palette_color[i]) + ';border:0px'
            self.palette_button[i].setStyleSheet(attr)

    def open_file(self):
        options = QFileDialog.Option.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", 
            "Images (*.jpg *.JPG *.jpeg *.png *.webp *.tiff *.tif *.bmp *.dib);;All Files (*)", 
            options=options
        )
        if not file_name:
            return
        self.Source = file_name
        resized = self.pixmap_open_img(5)
        
        # resized.scaledToHeight()
        
        self.image_label.setPixmap(resized)
        # rbf weights
        self.sample_weight_map = rbf_weights(self.means, self.sample_colors)

    def reset(self):
        resized = self.pixmap_open_img(self.K)
        # resized.scaledToHeight()
        self.image_label.setPixmap(resized)

    def save_file(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save File",
            "",
            "PNG (*.png);;JPG (*.jpg);;Images (*.jpg *.JPG *.jpeg *.png *.webp *.tiff *.tif *.bmp *.dib);;All Files (*)"
        )
        if len(file_name) == 0:
            return
        print('Saving to', file_name)
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.tiff', '.tif', '.bmp', '.dib')):
            file_name += '.png'
        self.img.save(file_name)
        print('Saved to', file_name)

    def set_number_of_palettes(self, text):
        self.K = int(text)
        for i in range(self.K, 7):
            attr = 'background-color:#EFEFEF;border:0px'
            self.palette_button[i].setStyleSheet(attr)
        self.calc_palettes(self.K)

    def UiComponents(self):
        self.main_layout = QVBoxLayout()

        Image_section = QWidget()
        image_section_layout = QHBoxLayout()
        self.image_label = QLabel()
        image_section_layout.addWidget(self.image_label)
        Color_wheel = QWidget()
        color_wheel_layout = QVBoxLayout()
        Color_wheel.setLayout(color_wheel_layout)

        self.palette_button = []
        for i in range(7):
            button = QPushButton()
            button.setStyleSheet('background-color:#EFEFEF;border:0px')
            button.clicked.connect(lambda _, i=i: self.clicked(i))
            self.palette_button.append(button)
            color_wheel_layout.addWidget(button)

        Button_section = QWidget()
        button_section_layout = QVBoxLayout()
        Button_section.setLayout(button_section_layout)

        OpenButton = QPushButton('Open')
        OpenButton.clicked.connect(self.open_file)
        button_section_layout.addWidget(OpenButton)

        SaveButton = QPushButton('Save')
        SaveButton.clicked.connect(self.save_file)
        button_section_layout.addWidget(SaveButton)

        AutoButton = QPushButton('Auto')
        AutoButton.clicked.connect(self.auto)
        button_section_layout.addWidget(AutoButton)

        ResetButton = QPushButton('Reset')
        ResetButton.clicked.connect(self.reset)
        button_section_layout.addWidget(ResetButton)

        StyleButton = QPushButton('Style')
        StyleButton.clicked.connect(self.style_transfer)
        button_section_layout.addWidget(StyleButton)

        Num_palettes = QComboBox()
        Num_palettes.addItems(['3', '4', '5', '6', '7'])
        Num_palettes.currentTextChanged.connect(self.set_number_of_palettes)
        button_section_layout.addWidget(Num_palettes)

        self.main_layout.addWidget(Color_wheel)
        self.main_layout.addWidget(Button_section)
        self.main_layout.addWidget(Image_section)
        self.setLayout(self.main_layout)

if __name__ == '__main__':
    app = QApplication([])
    window = Window()
    app.exec()

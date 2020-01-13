#!/usr/local/bin/python
# -*- coding: utf-8 -*-

# basic imports
import io
import os

import pandas as pd

# plotting imports
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
import seaborn as sns
import wordcloud
# WARNING: I had to modify the source code of stopwords in wordcloud.py file

# stats imports
import numpy as np
from scipy.stats import kruskal
import scikit_posthocs

# PyQt imports
from PyQt5 import QtCore, QtGui, QtWidgets, __file__

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(420, 600)
        pyqt_dir = os.path.dirname(__file__)
        QtWidgets.QApplication.addLibraryPath(os.path.join(pyqt_dir, "plugins"))
        MainWindow.setWindowIcon(QtGui.QIcon('plugins/kimono.png'))

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.gridLayout_3 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName("gridLayout_3")

        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")

        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout = QtWidgets.QGridLayout(self.tab)
        self.gridLayout.setObjectName("gridLayout")
        self.label_info_filetype = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setPointSize(10)

        self.label_info_filetype.setFont(font)
        self.label_info_filetype.setObjectName("label_info_filetype")
        self.gridLayout.addWidget(self.label_info_filetype, 0, 1, 1, 1)
        self.load_file_button = QtWidgets.QPushButton(self.tab)
        self.load_file_button.setObjectName("load_file_button")
        self.gridLayout.addWidget(self.load_file_button, 0, 0, 1, 1)
        self.load_file_button.clicked.connect(self.open_file)

        self.frame1 = QtWidgets.QFrame(self.tab)
        self.frame1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame1.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame1.setLineWidth(2)
        self.frame1.setMidLineWidth(1)
        self.frame1.setObjectName("frame1")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame1)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label1 = QtWidgets.QLabel(self.frame1)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label1.setFont(font)
        self.label1.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTop|QtCore.Qt.AlignTrailing)
        self.label1.setObjectName("label1")
        self.gridLayout_4.addWidget(self.label1, 0, 0, 1, 1)
        self.label1_opened_file = QtWidgets.QLabel(self.frame1)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label1_opened_file.setFont(font)
        self.label1_opened_file.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label1_opened_file.setWordWrap(True)
        self.label1_opened_file.setObjectName("label1_opened_file")
        self.gridLayout_4.addWidget(self.label1_opened_file, 0, 1, 1, 1)

        self.label_basic_stats = QtWidgets.QLabel(self.frame1)
        self.label_basic_stats.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_basic_stats.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_basic_stats.setWordWrap(True)
        self.label_basic_stats.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
        self.label_basic_stats.setObjectName("label_basic_stats")

        self.gridLayout.addWidget(self.frame1, 2, 0, 1, 2)

        self.scroll = QtWidgets.QScrollArea(self.frame1)
        self.gridLayout_4.addWidget(self.scroll, 1, 0, 3, 2)
        self.scroll.setWidgetResizable(True)
        scrollContent = QtWidgets.QWidget(self.scroll)
        scrollLayout = QtWidgets.QVBoxLayout(scrollContent)
        scrollContent.setLayout(scrollLayout)
        scrollLayout.addWidget(self.label_basic_stats)
        self.scroll.setWidget(scrollContent)

        self.tabWidget.addTab(self.tab, "")

        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_q_which_cols = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_q_which_cols.setFont(font)
        self.label_q_which_cols.setObjectName("label_q_which_cols")
        self.gridLayout_2.addWidget(self.label_q_which_cols, 3, 0, 1, 4)

        self.label_info_filetype_2 = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_info_filetype_2.setFont(font)
        self.label_info_filetype_2.setObjectName("label_info_filetype_2")
        self.gridLayout_2.addWidget(self.label_info_filetype_2, 1, 1, 1, 3)

        self.label_q_which_polarity = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_q_which_polarity.setFont(font)
        self.label_q_which_polarity.setObjectName("label_q_which_polarity")
        self.gridLayout_2.addWidget(self.label_q_which_polarity, 0, 0, 1, 3)

        self.clean_data_button = QtWidgets.QPushButton(self.tab_2)
        self.clean_data_button.setObjectName("clean_data_button")
        self.gridLayout_2.addWidget(self.clean_data_button, 1, 0, 1, 1)
        self.clean_data_button.clicked.connect(self.clean_data_function)

        self.comboBox_list_of_cols_polarity = QtWidgets.QComboBox(self.tab_2)
        self.comboBox_list_of_cols_polarity.setObjectName("comboBox_list_of_cols_polarity")
        self.comboBox_list_of_cols_polarity.addItem("")
        self.gridLayout_2.addWidget(self.comboBox_list_of_cols_polarity, 0, 3, 1, 1)


        self.label_x_axis = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_x_axis.setFont(font)
        self.label_x_axis.setObjectName("label_x_axis")
        self.gridLayout_2.addWidget(self.label_x_axis, 4, 0, 1, 2)
        self.comboBox_list_of_cols_x = QtWidgets.QComboBox(self.tab_2)
        self.comboBox_list_of_cols_x.setObjectName("comboBox_list_of_cols_x")
        self.comboBox_list_of_cols_x.addItem("")
        self.gridLayout_2.addWidget(self.comboBox_list_of_cols_x, 5, 0, 1, 2)

        self.comboBox_list_of_cols_y = QtWidgets.QComboBox(self.tab_2)
        self.comboBox_list_of_cols_y.setObjectName("comboBox_list_of_cols_y")
        self.comboBox_list_of_cols_y.addItem("")
        self.gridLayout_2.addWidget(self.comboBox_list_of_cols_y, 5, 2, 1, 2)
        self.label_2 = QtWidgets.QLabel(self.tab_2)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 6, 0, 1, 1)
        self.label_y_axis = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_y_axis.setFont(font)
        self.label_y_axis.setObjectName("label_y_axis")
        self.gridLayout_2.addWidget(self.label_y_axis, 4, 2, 1, 2)

        self.label_3 = QtWidgets.QLabel(self.tab_2)
        self.label_3.setText("Type of the graph:")
        self.label_3.setObjectName("label_3")
        self.label_3.setAlignment(QtCore.Qt.AlignRight)
        self.gridLayout_2.addWidget(self.label_3, 7, 0, 1, 2)

        self.comboBox_list_type_of_graph = QtWidgets.QComboBox(self.tab_2)
        self.comboBox_list_type_of_graph.setObjectName("comboBox_list_type_of_graph")
        self.comboBox_list_type_of_graph.addItem("Choose type")
        self.comboBox_list_type_of_graph.addItem("pointplot")
        self.comboBox_list_type_of_graph.addItem("stripplot")
        self.gridLayout_2.addWidget(self.comboBox_list_type_of_graph, 7, 2, 1, 2)

        self.label_5 = QtWidgets.QLabel(self.tab_2)
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 8, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.tab_2)
        self.label.setText("")
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 2, 0, 1, 1)

        self.pushButton_visualise_stats = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_visualise_stats.setObjectName("pushButton_visualise_stats")
        self.gridLayout_2.addWidget(self.pushButton_visualise_stats, 10, 0, 1, 4)

        self.pushButton_visualise_columns = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_visualise_columns.setObjectName("pushButton_visualise_columns")
        self.gridLayout_2.addWidget(self.pushButton_visualise_columns, 9, 0, 1, 4)

        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.tab_3)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label1_2 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label1_2.setFont(font)
        self.label1_2.setObjectName("label1_2")
        self.gridLayout_5.addWidget(self.label1_2, 0, 0, 1, 1)
        self.label1_opened_file_2 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label1_opened_file_2.setFont(font)
        self.label1_opened_file_2.setObjectName("label1_opened_file_2")
        self.label1_opened_file_2.setWordWrap(True)
        self.gridLayout_5.addWidget(self.label1_opened_file_2, 0, 1, 1, 1)
        self.comboBox_list_of_cols_tags_cloud = QtWidgets.QComboBox(self.tab_3)
        self.comboBox_list_of_cols_tags_cloud.setObjectName("comboBox_list_of_cols_tags_cloud")
        self.comboBox_list_of_cols_tags_cloud.addItem("")
        self.gridLayout_5.addWidget(self.comboBox_list_of_cols_tags_cloud, 1, 2, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.tab_3)
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        self.gridLayout_5.addWidget(self.label_7, 3, 0, 1, 1)

        self.pushButton_visualise_cloud = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_visualise_cloud.setObjectName("pushButton_visualise_cloud")

        self.btn_grp = QtWidgets.QButtonGroup()
        self.btn_grp.setExclusive(True)
        self.btn_grp.addButton(self.pushButton_visualise_columns)
        self.btn_grp.addButton(self.pushButton_visualise_stats)
        self.btn_grp.addButton(self.pushButton_visualise_cloud)
        self.btn_grp.buttonClicked.connect(self.visualise_or_stats)

        self.gridLayout_5.addWidget(self.pushButton_visualise_cloud, 2, 0, 1, 3)
        self.label_q_which_polarity_2 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_q_which_polarity_2.setFont(font)
        self.label_q_which_polarity_2.setObjectName("label_q_which_polarity_2")
        self.gridLayout_5.addWidget(self.label_q_which_polarity_2, 1, 0, 1, 2)
        self.label_8 = QtWidgets.QLabel(self.tab_3)
        self.label_8.setText("")
        self.label_8.setObjectName("label_8")
        self.gridLayout_5.addWidget(self.label_8, 4, 0, 1, 1)
        self.tabWidget.addTab(self.tab_3, "")
        self.gridLayout_3.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 404, 18))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionAbout.triggered.connect(self.about_window)

        self.actionOpen_file_csv_or_xlsx = QtWidgets.QAction(MainWindow)
        self.actionOpen_file_csv_or_xlsx.setObjectName("actionOpen_file_csv_or_xlsx")
        self.actionOpen_file_csv_or_xlsx.triggered.connect(self.open_file)

        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.menuFile.addAction(self.actionOpen_file_csv_or_xlsx)
        self.menuFile.addAction(self.actionExit)
        self.actionExit.triggered.connect(self.exit_check)

        self.menuHelp.addAction(self.actionAbout)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Fighting Without Fighting, ver. 1.0.0"))
        self.label_info_filetype.setText(_translate("MainWindow", "(must be a csv or Excel file)"))
        self.load_file_button.setText(_translate("MainWindow", "Load file..."))
        self.label1.setText(_translate("MainWindow", "You have opened the file:\n(loading can take a few seconds...)"))
        self.label1_opened_file.setText(_translate("MainWindow", ".............."))
        self.label_basic_stats.setText(_translate("MainWindow", "Your database basic info:..........."))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Load file and basic info"))
        self.label_q_which_cols.setText(_translate("MainWindow", "Which of your database columns would you like " +
        "to use for creating visualisations / statistics?"))
        self.label_x_axis.setText(_translate("MainWindow", "x axis:\n(categorical values)\n(e.g. positive, negative, neutral...)"))
        self.comboBox_list_of_cols_x.setItemText(0, _translate("MainWindow", "List of columns"))
        self.label_info_filetype_2.setText(_translate("MainWindow", "basic preprocessing: remove empty rows etc."))
        self.label_q_which_polarity.setText(_translate("MainWindow", "Which of your database columns is a \"Polarity\" column?"))
        self.clean_data_button.setText(_translate("MainWindow", "Clean data..."))
        self.comboBox_list_of_cols_polarity.setItemText(0, _translate("MainWindow", "List of columns"))
        self.comboBox_list_of_cols_y.setItemText(0, _translate("MainWindow", "List of columns"))
        self.label_y_axis.setText(_translate("MainWindow", "y axis:\n(continuous values)\n(e.g. 1,2,3,4...100)"))
        self.pushButton_visualise_stats.setText(_translate("MainWindow", "Statistical analysis"))
        self.pushButton_visualise_columns.setText(_translate("MainWindow", "Visualise a graph!"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Sentiment analysis"))
        self.label1_2.setText(_translate("MainWindow", "You have opened the file:"))
        self.label1_opened_file_2.setText(_translate("MainWindow", ".............."))
        self.comboBox_list_of_cols_tags_cloud.setItemText(0, _translate("MainWindow", "List of columns"))
        self.pushButton_visualise_cloud.setText(_translate("MainWindow", "Visualise a cloud!"))
        self.label_q_which_polarity_2.setText(_translate("MainWindow", "Column with text to be changed to a Tags Cloud:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Tags Cloud"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionAbout.setText(_translate("MainWindow", "About..."))
        self.actionOpen_file_csv_or_xlsx.setText(_translate("MainWindow", "Open file..."))
        self.actionExit.setText(_translate("MainWindow", "Exit"))

        Ui_MainWindow.polarity_col = self.comboBox_list_of_cols_polarity.currentText()

    def exit_check(self):
        sys.exit()

    def about_window(self):
        about_win = QtWidgets.QMessageBox()
        about_win.setWindowIcon(QtGui.QIcon('plugins/kimono.png'))
        about_win.setWindowTitle("About FWF")
        about_win.setText("You can find the full documentation here:\n\n"
        "https://github.com/leontikos/FWF\n\n" +
        "If you use this software, please quote us:\n" +
        "Ciechanowski L., Jemielniak D., Gloor P. (forthcoming). TUTORIAL: AI Research Without Coding - " +
        "The Art of Fighting Without Fighting.\nJournal of Business Research.")
        # TODO (maybe): add italics
        about_win.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
        about_win.exec_()

    def open_file(self):
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None,"Load file (csv or xlsx or xls)...",
            "","All Files (*);;Python Files (*.py)", options=options)
            if any(x in fileName for x in ['.xlsx', '.xls', '.csv']):
                self.label1_opened_file.setText(fileName.split('/')[-1])
                self.label1_opened_file_2.setText(fileName.split('/')[-1])
                self.basic_statistics(fileName)
                self.label_info_filetype_2.setText("basic preprocessing: remove empty rows etc.")
            else:
                error_message = QtWidgets.QErrorMessage()
                error_message.setWindowTitle("Wrong file type")
                error_message.showMessage("You can load only csv or xlsx or xls files")
                error_message.exec_()

    def basic_statistics(self, fileName):
        global df
        if fileName.split('.')[-1] == 'xls':
            df = pd.read_excel(fileName)
            df = df.reset_index(drop=True)
        elif fileName.split('.')[-1] == 'xlsx':
            df = pd.read_excel(fileName)
            df = df.reset_index(drop=True)
        elif fileName.split('.')[-1] == 'csv':
            df = pd.read_csv(fileName)
            df = df.reset_index(drop=True)
        # TODO (maybe): pick excel worksheet

        if any(x in fileName for x in ['.xlsx', '.xls', '.csv']):
            Ui_MainWindow.number_of_rows = len(df.index)
            buffer = io.StringIO()
            df.info(buf=buffer)
            df_info = buffer.getvalue()
            df_info = df_info.replace("<class 'pandas.core.frame.DataFrame'>", "")
            df_info = df_info.replace("RangeIndex:", "Rows:")
            df_info = df_info.replace("Data columns", "Column names")

            self.label_basic_stats.setText('Basic info:\n\n' +
            'Number of rows: {}\n\n'.format(Ui_MainWindow.number_of_rows) +
            'Number of columns: {}\n\n'.format(len(df.columns)) +
            df_info)
            self.comboBoxes_lists(df)

        # TODO: add a scrollable area
        # TODO: IMPORTANT add the list of categorical and continuous columns


    def comboBoxes_lists(self, df):
        self.comboBox_list_of_cols_polarity.clear()
        self.comboBox_list_of_cols_tags_cloud.clear()
        self.comboBox_list_of_cols_x.clear()
        self.comboBox_list_of_cols_y.clear()
        for col in df.columns.values:
            self.comboBox_list_of_cols_polarity.addItem(col)
            self.comboBox_list_of_cols_tags_cloud.addItem(col)
            self.comboBox_list_of_cols_x.addItem(col)
            self.comboBox_list_of_cols_y.addItem(col)

    def clean_data_function(self):
        Ui_MainWindow.polarity_col = self.comboBox_list_of_cols_polarity.currentText()

        global df
        if (Ui_MainWindow.polarity_col in df.columns) and ('polarity' in str.lower(Ui_MainWindow.polarity_col)):
            df = df.dropna(how='any', subset=[Ui_MainWindow.polarity_col])
            df = df[df.loc[:, Ui_MainWindow.polarity_col] != 'NONE'].reset_index(drop=True)
        else:
            df = df.dropna().reset_index(drop=True)

        self.label_info_filetype_2.setText("****** DONE + basic info changed ******")

        buffer = io.StringIO()
        df.info(buf=buffer)
        df_info = buffer.getvalue()
        df_info = df_info.replace("<class 'pandas.core.frame.DataFrame'>", "")
        df_info = df_info.replace("RangeIndex:", "Rows:")
        df_info = df_info.replace("Data columns", "Column names")

        self.label_basic_stats.setText('Basic info:\n\n' +
        'Number of rows (OLD file): {}\n\n'.format(Ui_MainWindow.number_of_rows) +
        'Number of rows AFTER CLEANING: {}\n\n'.format(len(df.index)) +
        # 'Polarity column values: {}\n\n'.format(df.loc[:, Ui_MainWindow.polarity_col].unique()) +
        'Number of columns: {}\n\n'.format(len(df.columns)) +
        df_info)

        # TODO: too much text does not fit the window

    def visualise_or_stats(self, btn):
        Ui_MainWindow.clicked_button = btn.text()
        Ui_MainWindow.text_for_cloud_col = self.comboBox_list_of_cols_tags_cloud.currentText()
        Ui_MainWindow.graph_type = self.comboBox_list_type_of_graph.currentText()
        Ui_MainWindow.x_col = self.comboBox_list_of_cols_x.currentText()
        Ui_MainWindow.y_col = self.comboBox_list_of_cols_y.currentText()
        self.SW = SecondWindow()
        self.SW.resize(800,600)
        self.SW.show()


class SecondWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
         super(SecondWindow, self).__init__()
         self.main_widget = QtWidgets.QWidget()
         self.setCentralWidget(self.main_widget)
         layout = QtWidgets.QVBoxLayout(self.main_widget)
         self.setWindowIcon(QtGui.QIcon('plugins/kimono.png'))

         if Ui_MainWindow.clicked_button == 'Visualise a cloud!':
             plotting_widget = PlottingClass(self.main_widget, width = 300, height = 300)
             layout.addWidget(plotting_widget)
             self.setWindowTitle("Fighting Without Fighting, ver. 1.0.0 - Tags Cloud")
         elif Ui_MainWindow.clicked_button == 'Visualise a graph!':
             plotting_widget = PlottingClass(self.main_widget, width = 300, height = 300)
             layout.addWidget(plotting_widget)
             self.setWindowTitle("Fighting Without Fighting, ver. 1.0.0 - Graphs")

         elif Ui_MainWindow.clicked_button == 'Statistical analysis':
             self.setWindowTitle("Fighting Without Fighting, ver. 1.0.0 - Statistics")

             self.frame_stats = QtWidgets.QFrame(self.main_widget, width = 300, height = 300)
             layout.addWidget(self.frame_stats)
             self.frame_stats.setFrameShape(QtWidgets.QFrame.StyledPanel)
             self.frame_stats.setFrameShadow(QtWidgets.QFrame.Plain)
             self.frame_stats.setLineWidth(2)
             self.frame_stats.setMidLineWidth(1)
             self.frame_stats.setObjectName("frame_stats")

             self.gridLayout_frame_stats = QtWidgets.QGridLayout(self.frame_stats)
             self.gridLayout_frame_stats.setObjectName("gridLayout_frame_stats")

             self.label_adv_stats = QtWidgets.QLabel(self.frame_stats)
             self.label_adv_stats.setFrameShape(QtWidgets.QFrame.StyledPanel)
             self.label_adv_stats.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
             self.label_adv_stats.setWordWrap(True)
             self.label_adv_stats.setObjectName("label_adv_stats")
             self.gridLayout_frame_stats.addWidget(self.label_adv_stats, 0, 0, 3, 2)
             self.label_adv_stats.setText("Advanced statistics:\n\n")
             self.label_adv_stats.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)

             self.tableDF = QtWidgets.QTableWidget(self.frame_stats)
             self.tableDF.setObjectName("tableDF")
             self.tableDF.setColumnCount(0)
             self.tableDF.setRowCount(0)
             self.tableDF.horizontalHeader().setVisible(True)
             self.tableDF.verticalHeader().setVisible(True)
             self.gridLayout_frame_stats.addWidget(self.tableDF, 3, 0, 3, 2)

             self.compute_stats()


    def compute_stats(self):
        global df

        # TODO: add a reset button to load the original database
        # TODO IMPORTANT: add a scrollable area

        df = df.dropna(how='any', subset=[Ui_MainWindow.x_col, Ui_MainWindow.y_col])

        stats_text = ("Advanced statistics:\n\n\n" +
        "Please mind that in order to carry out the statistical analysis, " +
        "we had to remove the empty rows for your chosen x axis and y axis columns!\n\n" +
        "Your initial data - number of rows: {}\n".format(Ui_MainWindow.number_of_rows) +
        "Your cleaned data - number of rows: {}\n\n".format(len(df.index)))

        # TODO: add number of elements in each category

        data = []
        for idx, val in enumerate(df.loc[:, Ui_MainWindow.x_col].unique()):
            data.insert(idx, df[df.loc[:, Ui_MainWindow.x_col] == val].loc[:, Ui_MainWindow.y_col])
        stat, p = kruskal(*data)

        stats_text = (stats_text +
        "Kruskal nonparametric test of variables:\n" +
        "{}   &   {}\n".format(Ui_MainWindow.x_col, Ui_MainWindow.y_col) +
        "Statistics={:.4f}, p={:.4f}\n".format(stat, p)
        )

        # interpret
        alpha = 0.05
        if p > alpha:
            significance_text = 'Same distributions (fail to reject H0) = There are NO significant differences between the groups.'
            stats_text = (stats_text + significance_text)
        else:
            significance_text = 'Different distributions (reject H0) = There are significant differences between the groups.'

            stats_text = (stats_text + significance_text +
            "\n\nPosthoc tests matrix, using a step-down method with Bonferroni adjustments:\n" +
            "(Each value indicates a p-value between each of the categories)"
            )

            posthocs_df = scikit_posthocs.posthoc_mannwhitney(df, val_col=Ui_MainWindow.y_col, group_col=Ui_MainWindow.x_col,
            p_adjust = 'holm')
            posthocs_df = posthocs_df.round(decimals=5)

            self.tableDF.setColumnCount(len(posthocs_df.columns))
            self.tableDF.setRowCount(len(posthocs_df.index))
            for i in range(len(posthocs_df.index)):
                for j in range(len(posthocs_df.columns)):
                    self.tableDF.setItem(i, j, QtWidgets.QTableWidgetItem(str(posthocs_df.iloc[i, j])))
            self.tableDF.setHorizontalHeaderLabels(posthocs_df.columns.values)
            self.tableDF.setVerticalHeaderLabels(posthocs_df.index.values)
            self.tableDF.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

        self.label_adv_stats.setText(stats_text)
        # TODO: add save button (maybe will be dealt with NavigationToolbar2QT)


class PlottingClass(FigureCanvas, Ui_MainWindow):
    def __init__(self, parent=None, width= 300, height= 300):
        fig = Figure(figsize=(width, height))
        self.axes = fig.add_subplot(111)

        if Ui_MainWindow.clicked_button == 'Visualise a cloud!':
            self.compute_cloud()
        elif Ui_MainWindow.clicked_button == 'Visualise a graph!':
            self.compute_graph()

        self.canvas = FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        # TODO: add NavigationToolbar2QT

    def compute_cloud(self):
        self.axes.clear()
        if Ui_MainWindow.text_for_cloud_col != 'List of columns':
            clean_text = df.loc[:, Ui_MainWindow.text_for_cloud_col].dropna()
            clean_text = [str(x) for x in clean_text]
            text = ''.join(clean_text)
            word_cloud = wordcloud.WordCloud(width=500, height=300, max_font_size=50,
                          max_words=100, background_color="white").generate(text)
            self.axes.imshow(word_cloud, interpolation="bilinear")
            self.axes.axis("off")
            # TODO: add category division choices (e.g. WordCloud for P+ polarity)
            # TODO: add palette change
            # TODO: add stopwords input

    def compute_graph(self):
        self.axes.clear()

        if Ui_MainWindow.polarity_col in Ui_MainWindow.x_col or Ui_MainWindow.polarity_col in Ui_MainWindow.y_col:
            order = ['P+', 'P', 'NEU', 'N', 'N+']
        else:
            order=None

        if Ui_MainWindow.graph_type == 'pointplot':
            sns.pointplot(x=df.loc[:, Ui_MainWindow.x_col], y=df.loc[:, Ui_MainWindow.y_col], order=order,
            ax=self.axes, dodge=.532, size = 20, join = False, palette="husl")
        elif Ui_MainWindow.graph_type == 'stripplot':
            sns.stripplot(x=df.loc[:, Ui_MainWindow.x_col], y=df.loc[:, Ui_MainWindow.y_col], order=order,
            ax=self.axes, dodge=True, jitter=True, size = 3, palette = "husl")
        # TODO: add pallette choice
        # TODO: add possibility of changing plots parameters
        # TODO: add option of cutting max values
        # TODO: FIXME: ADD WARNING notifications if columns not chosen


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

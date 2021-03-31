import sys, os, pickle
from PyQt5 import QtWidgets, QtCore
from uquake.core import event, UTCDateTime, read, read_events
from itertools import cycle

import numpy as np

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as \
    FigureCanvas
from matplotlib.figure import Figure
from matplotlib.transforms import offset_copy


class StreamPick(QtWidgets.QMainWindow):
    def __init__(self, stream_file, event_file):
        # Initialising QtWidgets
        QtCore.QLocale.setDefault(QtCore.QLocale.c())
        qApp = QtWidgets.QApplication(sys.argv)

        stream = read(stream_file)
        cat = read_events(event_file)

        # Init vars
        self.st = stream.copy()
        self._picks = []
        if cat is not None:
            for arrival in cat[0].preferred_origin().arrivals:
                self._picks.append(arrival.get_pick())
        self.save_file = None
        self.onset_types = ['emergent', 'impulsive', 'questionable']

        # Load filters from pickle
        try:
            self.bpfilter = pickle.load(open('.pick_filters', 'r'))
        except:
            self.bpfilter = []
        # Internal variables
        # Gui vars
        self._shortcuts = {'st_next': 'c',
                           'st_previous': 'x',
                           'filter_apply': 'f',
                           'pick_p': 'q',
                           'pick_s': 'w',
                           'pick_custom': 't',
                           'pick_remove': 'r',
                           }
        self._plt_drag = None
        self._current_filter = None
        # Init stations
        self._initStations()  # defines list self._stations
        self._stationCycle = cycle(self._stations)
        self._streamStation(self._stationCycle.next())
        # Init QtWidgets
        QtWidgets.QMainWindow.__init__(self)
        self.setup_ui()
        # exec QtApp
        qApp.exec_()
        
        self.main_widget = None
        self._wadati_plot = None

    def setup_ui(self):
        """
        setup the UI
        :return:
        """

        self.main_widget = QtWidgets.QWidget(self)
        # Init parts of the UI
        self._init_menu()
        self._create_status_bar()
        self._init_plots()

        # Define layout
        l = QtWidgets.QVBoxLayout(self.main_widget)
        l.addLayout(self.btnbar)
        l.addWidget(self.canvas)

        self.setCentralWidget(self.main_widget)
        self.setGeometry(300, 300, 1200, 800)
        self.setWindowTitle('obspy.core.Stream-Picker')
        self.show()

    def _init_plots(self):
        self.fig = Figure(facecolor='.86', dpi=72, frameon=True)
        # Change facecolor
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        # Draw the matplotlib figure
        self._draw_fig()
        # Connect the events
        self.fig.canvas.mpl_connect('scroll_event',
                                    self._plt_on_scroll)
        self.fig.canvas.mpl_connect('motion_notify_event',
                                    self._plt_on_drag)
        self.fig.canvas.mpl_connect('button_release_event',
                                    self._plt_on_button_release)
        self.fig.canvas.mpl_connect('button_press_event',
                                    self._plt_on_button_press)

    def _init_menu(self):
        # Next and Prev Button
        nxt = QtWidgets.QPushButton('Next >>',
                                shortcut=self._shortcuts['st_next'])
        nxt.clicked.connect(self._plt_next_station)
        nxt.setToolTip('shortcut <b>c</d>')
        nxt.setMaximumWidth(150)
        prv = QtWidgets.QPushButton('<< Prev',
                                shortcut=self._shortcuts['st_previous'])
        prv.clicked.connect(self._plt_prev_station)
        prv.setToolTip('shortcut <b>x</d>')
        prv.setMaximumWidth(150)

        # Stations drop-down
        self.stcb = QtWidgets.QComboBox(self)
        for st in self._stations:
            self.stcb.addItem(st)
        self.stcb.activated.connect(self._plt_station)
        self.stcb.setMaximumWidth(100)
        self.stcb.setMinimumWidth(80)

        # Filter buttons
        self.fltrbtn = QtWidgets.QPushButton('Filter Trace',
                                    shortcut=self._shortcuts['filter_apply'])
        self.fltrbtn.setToolTip('shortcut <b>f</b>')
        self.fltrbtn.setCheckable(True)
        #self.fltrbtn.setAutoFillBackground(True)
        self.fltrbtn.setStyleSheet(QtCore.QString(
                    'QPushButton:checked {background-color: lightgreen;}'))
        self.fltrbtn.clicked.connect(self._app_filter)

        self.fltrcb = QtWidgets.QComboBox(self)
        self.fltrcb.activated.connect(self._change_filter)
        self.fltrcb.setMaximumWidth(170)
        self.fltrcb.setMinimumWidth(150)
        self._update_filter_combo_box()  # fill QComboBox

        # edit/delete filer buttons
        fltredit = QtWidgets.QPushButton('Edit')
        fltredit.resize(fltredit.sizeHint())
        fltredit.clicked.connect(self._edit_filter)

        fltrdel = QtWidgets.QPushButton('Delete')
        fltrdel.resize(fltrdel.sizeHint())
        fltrdel.clicked.connect(self._delete_filter)

        btnstyle = QtWidgets.QFrame(fltredit)
        btnstyle.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        btnstyle = QtWidgets.QFrame(fltrdel)
        btnstyle.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)

        # onset type
        _radbtn = []
        for _o in self.onset_types:
                _radbtn.append(QtWidgets.QRadioButton(str(_o[0].upper())))
                _radbtn[-1].setToolTip('Onset ' + _o)
                _radbtn[-1].clicked.connect(self._draw_picks)
                if _o == 'impulsive':
                    _radbtn[-1].setChecked(True)
        self.onsetGrp = QtWidgets.QButtonGroup()
        self.onsetGrp.setExclusive(True)
        onsetbtns = QtWidgets.QHBoxLayout()
        for _i, _btn in enumerate(_radbtn):
            self.onsetGrp.addButton(_btn, _i)
            onsetbtns.addWidget(_btn)

        # Arrange buttons
        vline = QtWidgets.QFrame()
        vline.setFrameStyle(QtWidgets.QFrame.VLine | QtWidgets.QFrame.Raised)
        self.btnbar = QtWidgets.QHBoxLayout()
        self.btnbar.addWidget(prv)
        self.btnbar.addWidget(nxt)
        self.btnbar.addWidget(QtWidgets.QLabel('Station'))
        self.btnbar.addWidget(self.stcb)
        ##
        self.btnbar.addWidget(vline)
        self.btnbar.addWidget(self.fltrbtn)
        self.btnbar.addWidget(self.fltrcb)
        self.btnbar.addWidget(fltredit)
        self.btnbar.addWidget(fltrdel)
        ##
        self.btnbar.addWidget(vline)
        self.btnbar.addWidget(QtWidgets.QLabel('Pick Onset: '))
        self.btnbar.addLayout(onsetbtns)
        self.btnbar.addStretch(3)

        # Menubar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(QtWidgets.QIcon().fromTheme('document-save'),
                            'Save', self._save_catalog)
        fileMenu.addAction(QtWidgets.QIcon().fromTheme('document-save'),
                            'Save as QuakeML File', self._save_catalog_dialog)
        fileMenu.addAction(QtWidgets.QIcon().fromTheme('document-open'),
                            'Load QuakeML File', self._open_catalog_dialog)
        fileMenu.addSeparator()
        fileMenu.addAction('Save Plot', self._save_plot_dialog)
        fileMenu.addSeparator()
        fileMenu.addAction(QtWidgets.QIcon().fromTheme('application-exit'),
                            'Exit', self.close)
        #windowMenu = menubar.addMenu('&Windows')
        #windowMenu.addAction('Wadati Diagram', self._opnWadatiPlot)
        aboutMenu = menubar.addMenu('&About')
        aboutMenu.addAction(QtWidgets.QIcon().fromTheme('info'),
                            'Info', self._info_dialog)

    def _draw_fig(self):
        """
        Draws all matplotlib figures
        """
        num_plots = len(self._current_st)
        self.fig.clear()
        self._app_filter(draw=False)
        for _i, tr in enumerate(self._current_st):
            ax = self.fig.add_subplot(num_plots, 1, _i)
            ax.plot(tr.data, 'k', antialiased=True, rasterized=True, lod=False)
            ax.axhline(0, color='k', alpha=.05)
            ax.set_xlim([0, tr.data.size])
            ax.text(.02, .925, self._current_st[_i].id,
                        transform=ax.transAxes, va='top', ha='left', alpha=.75)
            ax.channel = tr.stats.channel
            if _i == 0:
                ax.set_xlabel('Seconds')

        # plot picks
        self._draw_picks(draw=False)
        self.fig.suptitle('%s - %s - %s / %.1f Hz / %d samples per chanel' % (self._current_st[-1].stats.network,
                            self._current_st[-1].stats.station,
                            self._current_st[-1].stats.starttime.isoformat(),
                            1./self._current_st[-1].stats.delta,
                            self._current_st[-1].stats.npts),
                            x=.2)
        self._update_status_bar()
        self._canvas_draw()

    def _initStations(self):
        """
        Creates a list holding unique station names
        """
        self._stations = []
        for _tr in self.st:
            if _tr.stats.station not in self._stations:
                self._stations.append(_tr.stats.station)
        self._stations.sort()

    def _getPhases(self):
        """
        Creates a list holding unique phase names
        """
        phases = []
        for _pick in self._picks:
            if _pick.phase_hint not in phases:
                phases.append(_pick.phase_hint)
        return phases

    def _streamStation(self, station):
        """
        Copies the current stream object from self.st through
        obspy.stream.select(station=)
        """
        if station not in self._stations:
            return
        self._current_st = self.st.select(station=station).copy()
        self._current_stname = station
        self._current_network = self._current_st[0].stats.network
        # Sort and detrend streams
        self._current_st.sort(['channel'])
        self._current_st.detrend('linear')

    def _setPick(self, xdata, phase, channel, polarity='undecideable', overwrite_existing=False):
        """
        Write obspy.core.event.Pick into self._picks list
        """
        picktime = self._current_st[0].stats.starttime +\
                (xdata * self._current_st[0].stats.delta)

        this_pick = event.Pick()
        overwrite = True
        # Overwrite existing phase's picktime
        if overwrite_existing:
            for _pick in self._get_picks():
                if _pick.phase_hint == phase and\
                        _pick.waveform_id.channel_code == channel:
                    this_pick = _pick
                    overwrite = False
                    break

        creation_info = event.CreationInfo(
            author='ObsPy.StreamPick',
            creation_time=UTCDateTime())
        # Create new event.Pick()
        this_pick.time = picktime
        this_pick.phase_hint = phase
        this_pick.waveform_id = event.WaveformStreamID(
            network_code=self._current_st[0].stats.network,
            station_code=self._current_st[0].stats.station,
            location_code=self._current_st[0].stats.location,
            channel_code=channel)
        this_pick.evaluation_mode = 'manual'
        this_pick.creation_info = creation_info
        this_pick.onset = self.onset_types[self.onsetGrp.checkedId()]
        this_pick.evaluation_status = 'preliminary'
        this_pick.polarity = polarity
        #if self._current_filter is not None:
        #    this_pick.comments.append(event.Comment(
        #                text=str(self.bpfilter[self.fltrcb.currentIndex()])))
        if overwrite:
            self._picks.append(this_pick)

    def _del_picks(self, network, station, channel):
        """
        Deletes pick from catalog
        """
        for _i, _pick in enumerate(self._picks):
            if _pick.waveform_id.network_code == network\
                    and _pick.waveform_id.station_code == station\
                    and _pick.waveform_id.channel_code == channel:
                self._picks.remove(_pick)

    def _get_picks(self):
        """
        Create a list of picks for the current plot
        """
        this_st_picks = []
        for _i, pick in enumerate(self._picks):
            if pick.waveform_id.station_code == self._current_stname and\
                    self._current_st[0].stats.starttime <\
                    pick.time < self._current_st[0].stats.endtime:
                this_st_picks.append(_i)
        return [self._picks[i] for i in this_st_picks]

    def _get_pick_position(self, picks):
        """
        Convert picktimes into relative positions along x-axis
        """
        xpicks = []
        for _pick in picks:
            xpicks.append((_pick.time-self._current_st[0].stats.starttime)
                            / self._current_st[0].stats.delta)
        return np.array(xpicks)

    def _draw_picks(self, draw=True):
        """
        Draw picklines onto axes
        """
        picks = self._get_picks()
        xpicks = self._get_pick_position(picks)

        for _ax in self.fig.get_axes():
            lines = []
            labels = []
            points = []
            transOffset = offset_copy(_ax.transData, fig=self.fig,
                            x=5, y=0, units='points')
            for _i, _xpick in enumerate(xpicks):
                if picks[_i].phase_hint == 'S':
                    color = 'r'
                elif picks[_i].phase_hint == 'P':
                    color = 'g'
                else:
                    color = 'b'
                if _ax.channel != picks[_i].waveform_id.channel_code:
                    alpha = .2
                else:
                    alpha = .8

                lines.append(matplotlib.lines.Line2D([_xpick, _xpick],
                            [_ax.get_ylim()[0]*.9, _ax.get_ylim()[1]*.8],
                            color=color, alpha=alpha, rasterized=True))
                lines[-1].obspy_pick = picks[_i]

                points.append(matplotlib.lines.Line2D([_xpick], [_ax.lines[0].get_ydata()[int(_xpick)]],
                            marker='o', mfc=color, mec=color, alpha=alpha, ms=5))

                labels.append(matplotlib.text.Text(_xpick,
                            _ax.get_ylim()[0]*.8, text=picks[_i].phase_hint,
                            color=color, size=10, alpha=alpha,
                            transform=transOffset))

            # delete all artists
            del _ax.artists[0:]
            # add updated objects
            for li, la, po in zip(lines, labels, points):
                _ax.add_artist(li)
                _ax.add_artist(la)
                _ax.add_artist(po)

        if draw:
            self._canvas_draw()

    # Plot Controls
    def _plt_on_scroll(self, event):
        """
        Scrolls/Redraws the plots along x axis
        """
        if event.inaxes is None:
            return

        if event.key == 'control':
            axes = [event.inaxes]
        else:
            axes = self.fig.get_axes()

        for _ax in axes:
            left = _ax.get_xlim()[0]
            right = _ax.get_xlim()[1]
            extent = right - left
            dzoom = .2 * extent
            aspect_left = (event.xdata - _ax.get_xlim()[0]) / extent
            aspect_right = (_ax.get_xlim()[1] - event.xdata) / extent

            if event.button == 'up':
                left += dzoom * aspect_left
                right -= dzoom * aspect_right
            elif event.button == 'down':
                left -= dzoom * aspect_left
                right += dzoom * aspect_right
            else:
                return
            _ax.set_xlim([left, right])
        self._canvas_draw()

    def _plt_on_drag(self, event):
        """
        Drags/Redraws the plot upon drag
        """
        if event.inaxes is None:
            return

        if event.key == 'control':
            axes = [event.inaxes]
        else:
            axes = self.fig.get_axes()

        if event.button == 2:
            if self._plt_drag is None:
                self._plt_drag = event.xdata
                return
            for _ax in axes:
                _ax.set_xlim([_ax.get_xlim()[0] +
                        (self._plt_drag - event.xdata),
                        _ax.get_xlim()[1] + (self._plt_drag - event.xdata)])
        else:
            return
        self._canvas_draw()

    def _plt_on_button_release(self, event):
        """
        On Button Release Reset drag variable
        """
        self._plt_drag = None

    def _plt_on_button_press(self, event):
        """
        This Function is evoked when the user picks
        """
        if event.key is not None:
            event.key = event.key.lower()
        if event.inaxes is None:
            return
        channel = event.inaxes.channel
        tr_amp = event.inaxes.lines[0].get_ydata()[int(event.xdata)+3] -\
                    event.inaxes.lines[0].get_ydata()[int(event.xdata)]
        if tr_amp < 0:
            polarity = 'negative'
        elif tr_amp > 0:
            polarity = 'positive'
        else:
            polarity = 'undecideable'

        if event.key == self._shortcuts['pick_p'] and event.button == 1:
            self._setPick(event.xdata, phase='P', channel=channel,
                            polarity=polarity)
        elif event.key == self._shortcuts['pick_s'] and event.button == 1:
            self._setPick(event.xdata, phase='S', channel=channel,
                            polarity=polarity)
        elif event.key == self._shortcuts['pick_custom'] and event.button == 1:
            text, ok = QtWidgets.QInputDialog.getItem(self, 'Custom Phase',
                'Enter phase name:', self._getPhases())
            if ok:
                self._setPick(event.xdata, phase=text, channel=channel,
                                polarity=polarity)
        elif event.key == self._shortcuts['pick_remove']:
            self._del_picks(network=self._current_network,
                            station=self._current_stname,
                            channel=channel)
        else:
            return
        self._update_status_bar()
        self._draw_picks()

    def _plt_next_station(self):
        """
        Plot next station
        """
        self._streamStation(self._stationCycle.next())
        self._draw_fig()

    def _plt_prev_station(self):
        """
        Plot previous station
        """
        for _i in range(len(self._stations)-1):
            prevStation = self._stationCycle.next()
        self._streamStation(prevStation)
        self._draw_fig()

    def _plt_station(self):
        """
        Plot station from DropDown Menu
        """
        _i = self.stcb.currentIndex()
        while self._stationCycle.next() != self._stations[_i]:
            pass
        self._streamStation(self._stations[_i])
        self._draw_fig()

    # Filter functions
    def _app_filter(self, button=True, draw=True):
        """
        Apply bandpass filter
        """
        _i = self.fltrcb.currentIndex()
        self._streamStation(self._current_stname)
        if self.fltrbtn.isChecked() is False:
            self._current_filter = None
        else:
            self._current_st.filter('bandpass',
                                    freqmin=self.bpfilter[_i]['freqmin'],
                                    freqmax=self.bpfilter[_i]['freqmax'],
                                    corners=self.bpfilter[_i]['corners'],
                                    zerophase=True)
            self._current_filter = _i
        for _i, _ax in enumerate(self.fig.get_axes()):
            if len(_ax.lines) == 0:
                continue
            _ax.lines[0].set_ydata(self._current_st[_i].data)
            _ax.relim()
            _ax.autoscale_view()
        if draw is True:
            self._draw_picks(draw=False)
            self._canvas_draw()
        self._update_status_bar()

    def _new_filter(self):
        """
        Create new filter
        """
        newFilter = self.DefineFilter(self)
        if newFilter.exec_():
                self.bpfilter.append(newFilter.getValues())
                self._update_filter_combo_box()
                self.fltrcb.setCurrentIndex(len(self.bpfilter)-1)
                self._app_filter()

    def _edit_filter(self):
        """
        Edit existing filter
        """
        _i = self.fltrcb.currentIndex()
        this_filter = self.bpfilter[_i]
        editFilter = self.DefineFilter(self, this_filter)
        if editFilter.exec_():
                self.bpfilter[_i] = editFilter.getValues()
                self._update_filter_combo_box()
                self.fltrcb.setCurrentIndex(_i)
                self._app_filter()

    def _delete_filter(self):
        """
        Delete filter
        """
        _i = self.fltrcb.currentIndex()
        self.fltrbtn.setChecked(False)
        self.bpfilter.pop(_i)
        self._update_filter_combo_box()
        self._app_filter()

    def _change_filter(self, index):
        """
        Evoke this is filter in drop-down is changed
        """
        if index == len(self.bpfilter):
            return self._new_filter()
        else:
            return self._app_filter()

    def _update_filter_combo_box(self):
        """
        Update the filter QComboBox
        """
        self.fltrcb.clear()
        self.fltrcb.setCurrentIndex(-1)
        for _i, _f in enumerate(self.bpfilter):
            self.fltrcb.addItem('%s [%.2f - %.2f Hz]' % (_f['name'],
                _f['freqmin'], _f['freqmax']))
        self.fltrcb.addItem('Create new Filter...')

    # Status bar functions
    def _create_status_bar(self):
        """
        Creates the status bar
        """
        sb = QtWidgets.QStatusBar()
        sb.setFixedHeight(18)
        self.setStatusBar(sb)
        self.statusBar().showMessage('Ready')

    def _update_status_bar(self, statustext=None):
        """
        Updates the statusbar text
        """
        if statustext is None:
            self.stcb.setCurrentIndex(
                self._stations.index(self._current_stname))
            msg = 'Station %i/%i - %i Picks' % (
                self._stations.index(self._current_stname)+1,
                len(self._stations), len(self._get_picks()))
            if self._current_filter is not None:
                msg += ' - Bandpass %s [%.2f - %.2f Hz]' % (
                    self.bpfilter[self._current_filter]['name'],
                    self.bpfilter[self._current_filter]['freqmin'],
                    self.bpfilter[self._current_filter]['freqmax'])
            else:
                msg += ' - Raw Data'
            self.statusBar().showMessage(msg)

    def _open_catalog_dialog(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self,
                        'Load QuakeML Picks',
                        os.getcwd(), 'QuakeML Format (*.xml)', '20')
        if filename:
            self._open_catalog(str(filename))
            self.save_file = str(filename)

    def _open_catalog(self, filename):
        """
        Open existing QuakeML catalog
        """
        try:
            print('Opening QuakeML Catalog %s' % filename)
            cat = event.readEvents(filename)
            self._picks = cat[0].picks
            self._draw_picks()
        except:
            msg = 'Could not open QuakeML file %s' % (filename)
            raise IOError(msg)

    def _save_catalog_dialog(self):
        """
        Save catalog through QtDialog
        """
        self.save_file = QtWidgets.QFileDialog.getSaveFileName(self,
                        'Save QuakeML Picks',
                                                           os.getcwd(), 'QuakeML Format (*.xml)')
        if not self.save_file:
            self.save_file = None
            return
        self.save_file = str(self.save_file)
        if os.path.splitext(self.save_file)[1].lower() != '.xml':
            self.save_file += '.xml'
        self._save_catalog()

    def _save_catalog(self, filename=None):
        """
        Saves the catalog to filename
        """
        if self.save_file is None and filename is None:
            return self._save_catalog_dialog()
        if filename is not None:
            savefile = filename
        else:
            savefile = self.save_file
        cat = event.Catalog()
        cat.events.append(event.Event(picks=self._picks))
        cat.write(savefile, format='QUAKEML')
        print('Picks saved as %s' % savefile)

    def _save_plot_dialog(self):
        """
        Save Plot Image Qt Dialog and Matplotlib wrapper
        """
        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Plot',
                        os.getcwd(),
                        'Image Format (*.png *.pdf *.ps *.svg *.eps)')
        if not filename:
            return
        filename = str(filename)
        format = os.path.splitext(filename)[1][1:].lower()
        if format not in ['png', 'pdf', 'ps', 'svg', 'eps']:
            format = 'png'
            filename += '.' + format
        self.fig.savefig(filename=filename, format=format, dpi=72)

    def get_picks(self):
        return self._picks

    def _open_wadati_plot(self):
        self._wadati_plot = QtWidgets.NewWindow()
        self._wadati_plot.show()

    def _info_dialog(self):
        msg = """
                <h3><b>obspy.core.stream-Picker</b></h3>
                <br><br>
                <div>
                StreamPick is a lightweight seismological
                wave time picker for <code>obspy.core.Stream()</code>
                objects. It further utilises the <code>obspy.core.event</code>
                class to store picks in the QuakeML format.
                </div>
                <h4>Controls:</h4>
                <blockquote>
                <table>
                    <tr>
                        <td width=20><b>%s</b></td><td>Next station</td>
                    </tr>
                    <tr>
                        <td width=20><b>%s</b></td><td>Previous station</td>
                    </tr>
                    <tr>
                        <td width=20><b>%s</b></td><td>Toggle filter</td>
                    </tr>
                    <tr>
                        <td width=20><b>%s</b></td>
                        <td>Set P-Phase pick at mouse position</td>
                    </tr>
                    <tr>
                        <td width=20><b>%s</b></td>
                        <td>Set S-Phase pick at mouse position</td>
                    </tr>
                    <tr>
                        <td width=20><b>%s</b></td>
                        <td>Set custom phase pick at mouse position</td>
                    </tr>
                    <tr>
                        <td width=20><b>%s</b></td>
                        <td>Remove last pick in trace</td>
                    </tr>
                </table>
                </blockquote>
                <h4>Plot Controls:</h4>
                <blockquote>
                Use mouse wheel to zoom in- and out. Middle mouse button moves
                plot along x-axis.<br>
                Hit <b>Ctrl</b> to manipulate a single plot.
                <br>
                </blockquote>
                <div>
                Programm stores filter parameters in <code>.pick_filter</code>
                and a backup of recent picks in
                <code>.picks-obspy.xml.bak</code>.<br><br>
                See <a href=http://www.github.org/miili/StreamPick>
                http://www.github.org/miili/StreamPick</a> and
                <a href=http://www.obspy.org>http://www.obspy.org</a>
                for further documentation.
                </div>
                """ % (
                    self._shortcuts['st_next'],
                    self._shortcuts['st_previous'],
                    self._shortcuts['filter_apply'],
                    self._shortcuts['pick_p'],
                    self._shortcuts['pick_s'],
                    self._shortcuts['pick_custom'],
                    self._shortcuts['pick_remove'],
                    )
        QtWidgets.QMessageBox.about(self, 'About', msg)

    def _canvas_draw(self):
        """
        Redraws the canvas and re-sets mouse focus
        """
        for _i, _ax in enumerate(self.fig.get_axes()):
            _ax.set_xticklabels(_ax.get_xticks() * self._current_st[_i].stats.delta)
        self.canvas.draw_idle()
        self.canvas.flush_events()
        self.canvas.setFocus()
        return

    def close_event(self, evnt):
        """
        This function is called upon closing the QtWidgets
        """
        # Save Picks
        pickle.dump(self.bpfilter, open('.pick_filters', 'w'))
        # Save Catalog
        if len(self._picks) > 0:
            self._save_catalog('.picks-obspy.xml.bak')
        if self.save_file is None and len(self._picks) > 0:
            ask = QtWidgets.QMessageBox.question(self, 'Save Picks?',
                'Do you want to save your picks?',
                QtWidgets.QMessageBox.Save |
                QtWidgets.QMessageBox.Discard |
                QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Save)
            if ask == QtWidgets.QMessageBox.Save:
                self._save_catalog()
            elif ask == QtWidgets.QMessageBox.Cancel:
                evnt.ignore()
        print(self._picks)


    # Filter Dialog
    class DefineFilter(QtWidgets.QDialog):
        def __init__(self, parent=None, filtervalues=None):
            """
            Bandpass filter dialog... Qt layout and stuff
            """
            QtWidgets.QDialog.__init__(self, parent)
            self.setWindowTitle('Create new Bandpass-Filter')

            # Frequency QDoubleSpinBoxes
            self.frqmin = QtWidgets.QDoubleSpinBox(decimals=2, maximum=100,
                            minimum=0.01, singleStep=0.1, value=0.1)
            self.frqmax = QtWidgets.QDoubleSpinBox(decimals=2, maximum=100,
                            minimum=0.01, singleStep=0.1, value=10.0)

            # Radio buttons for corners
            _corners = [2, 4, 8]
            _radbtn = []
            for _c in _corners:
                _radbtn.append(QtWidgets.QRadioButton(str(_c)))
                if _c == 4:
                    _radbtn[-1].setChecked(True)

            self.corner = QtWidgets.QButtonGroup()
            self.corner.setExclusive(True)

            radiogrp = QtWidgets.QHBoxLayout()
            for _i, _r in enumerate(_radbtn):
                self.corner.addButton(_r, _corners[_i])
                radiogrp.addWidget(_radbtn[_i])

            # Filter name
            self.fltname = QtWidgets.QLineEdit('Filter Name')
            self.fltname.selectAll()

            # Make Layout
            grid = QtWidgets.QGridLayout()
            grid.addWidget(QtWidgets.QLabel('Filter Name'), 0, 0)
            grid.addWidget(self.fltname, 0, 1)
            grid.addWidget(QtWidgets.QLabel('Min. Frequency'), 1, 0)
            grid.addWidget(self.frqmin, 1, 1)
            grid.addWidget(QtWidgets.QLabel('Max. Frequency'), 2, 0)
            grid.addWidget(self.frqmax, 2, 1)
            grid.addWidget(QtWidgets.QLabel('Corners'), 3, 0)
            grid.addLayout(radiogrp, 3, 1)
            grid.setVerticalSpacing(10)

            btnbox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                            QtWidgets.QDialogButtonBox.Cancel)
            btnbox.accepted.connect(self.accept)
            btnbox.rejected.connect(self.reject)

            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(QtWidgets.QLabel('Define a minimum and maximum' +
                ' frequency\nfor the bandpass filter.\nFunction utilises ' +
                'obspy.signal.filter (zerophase=True).\n'))
            layout.addLayout(grid)
            layout.addWidget(btnbox)

            if filtervalues is not None:
                self.fltname.setText(filtervalues['name'])
                self.frqmin.setValue(filtervalues['freqmin'])
                self.frqmax.setValue(filtervalues['freqmax'])
                self.corner.button(filtervalues['corners']).setChecked(True)

            self.setLayout(layout)
            self.setSizeGripEnabled(False)

        def getValues(self):
            """
            Return filter dialogs values as a dictionary
            """
            return dict(name=str(self.fltname.text()),
                        freqmin=float(self.frqmin.cleanText()),
                        freqmax=float(self.frqmax.cleanText()),
                        corners=int(int(self.corner.checkedId())))



#st = read('../OKAS01/*.mseed')
#for tr in st:
#    tr.trim(starttime=tr.stats.starttime, endtime=tr.stats.starttime+60)
#new_pick = streamPick(stream=st)
#print(new_pick.picks)
"""
src/ui/laptime_map_picker.py
Picker dialog that lists all laptime-map .npz files found in a directory.
Styled consistently with the rest of the app (global stylesheet, same widget
conventions as LoadOptimizationDialog).
"""

import os
import glob

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QMessageBox
)
from PyQt6.QtCore import Qt


class LaptimeMapPickerDialog(QDialog):

    def __init__(self, lapmaps_dir: str, parent=None):
        super().__init__(parent)
        self._lapmaps_dir = lapmaps_dir
        self._filepath: str | None = None

        self.setWindowTitle("Select Laptime Map")
        self.resize(560, 360)

        self._build_ui()
        self._populate()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # table (same pattern as LoadOptimizationDialog)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["File name", "Size", "Directory"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemDoubleClicked.connect(self._on_load)
        layout.addWidget(self.table)

        # empty-state label shown when no files exist
        self._lbl_empty = QLabel(
            "No laptime maps found.\n"
            "Generate one first with the external sweep script."
        )
        self._lbl_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_empty.hide()
        layout.addWidget(self._lbl_empty)

        # button row – matching LoadOptimizationDialog order (load | cancel)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self._btn_load = QPushButton("Load selected")
        self._btn_load.setObjectName("BtnRunSingle")
        self._btn_load.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_load.clicked.connect(self._on_load)
        self._btn_load.setEnabled(False)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_cancel.clicked.connect(self.reject)

        btn_row.addStretch()
        btn_row.addWidget(self._btn_load)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        self.table.itemSelectionChanged.connect(
            lambda: self._btn_load.setEnabled(len(self.table.selectedItems()) > 0)
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Data
    # ─────────────────────────────────────────────────────────────────────────

    def _populate(self):
        d = self._lapmaps_dir

        if not os.path.isdir(d):
            self._show_empty()
            return

        files = sorted(glob.glob(os.path.join(d, "*.npz")))
        if not files:
            self._show_empty()
            return

        self._files_map: dict[int, str] = {}
        for row, fp in enumerate(files):
            name = os.path.basename(fp)
            size_kb = os.path.getsize(fp) / 1024.0
            dir_label = os.path.relpath(os.path.dirname(fp))

            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(name))

            size_item = QTableWidgetItem(f"{size_kb:.1f} kB")
            size_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 1, size_item)

            self.table.setItem(row, 2, QTableWidgetItem(dir_label))
            self._files_map[row] = fp

        # select first row by default
        self.table.selectRow(0)

    def _show_empty(self):
        self.table.hide()
        self._lbl_empty.show()
        self._btn_load.setEnabled(False)

    # ─────────────────────────────────────────────────────────────────────────
    # Actions
    # ─────────────────────────────────────────────────────────────────────────

    def _on_load(self):
        selected = self.table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select a laptime map to load.")
            return
        row = selected[0].row()
        self._filepath = self._files_map.get(row)
        self.accept()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def selected_filepath(self) -> str | None:
        """Returns the full path of the chosen .npz, or None if nothing selected."""
        return self._filepath

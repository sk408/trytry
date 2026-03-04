"""Tools tab — Admin + Snapshots combined with internal sub-tabs."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from src.ui.views.admin_view import AdminView
from src.ui.views.snapshots_view import SnapshotsView


class ToolsView(QWidget):
    """Wrapper with sub-tabs for Admin and Snapshots."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.sub_tabs = QTabWidget()
        self.sub_tabs.setDocumentMode(True)

        self.admin = AdminView(parent)
        self.snapshots = SnapshotsView(parent)

        self.sub_tabs.addTab(self.admin, "Admin")
        self.sub_tabs.addTab(self.snapshots, "Snapshots")

        layout.addWidget(self.sub_tabs)

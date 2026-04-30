from __future__ import annotations

import io
import logging
import time
from collections.abc import Callable
from enum import Enum
from pathlib import Path

import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from PIL import Image
from traitlets import Bunch

from sar_pattern_validation.sample_catalog import (
    DatabaseSampleCatalog,
    DatabaseSampleColumn,
    DatabaseSampleFilterOption,
    DatabaseSampleFilters,
)

from .models import UiState, WorkflowResultPayload
from .runner import SarPatternValidationRunner
from .runtime import (
    WorkspacePaths,
    default_workspace_paths,
    ensure_notebook_prerequisites,
)
from .state import load_or_migrate_ui_state, save_ui_state

TRANSPARENT_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00"
    b"\x00\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
)


class GuiColors(str, Enum):
    PRIMARY = "#0090D0"
    WHITE = "#FFFFFF"
    LOADING = "#566670"
    FAIL = "#9B2423"
    TEXT_PRIMARY = "#FFFFFF"


class ResultTableRow(str, Enum):
    SSAR = "sSAR [W/kg]"


class ResultTableColumn(str, Enum):
    REF_30_DBM = "Reference 30 dBm"
    MEASURED_30_DBM = "Measured 30 dBm"
    SCALING_ERROR = "Scaling Error [%]"


class OutputWidgetHandler(logging.Handler):
    """Custom logging handler sending logs to an output widget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out = widgets.Output(
            layout={
                "width": "90%",
                "height": "600px",
                "border": "1px solid black",
                "overflow": "scroll hidden",
                "flex_flow": "column",
                "display": "flex",
            }
        )

    def emit(self, record) -> None:
        formatted_record = self.format(record)
        new_output = {
            "name": "stdout",
            "output_type": "stream",
            "text": formatted_record + "\n",
        }
        self.out.outputs = (new_output,) + self.out.outputs

    def show_logs(self) -> widgets.Output:
        return self.out


def resample_colorbar_to_match_plot_inplace(
    colorbar_path: Path, plot_path: Path
) -> None:
    with Image.open(plot_path) as plot_img:
        target_height = plot_img.size[1]
    with Image.open(colorbar_path) as colorbar_img:
        colorbar_img = colorbar_img.convert("RGBA")
        cb_width, cb_height = colorbar_img.size
        if cb_height == 0:
            raise ValueError("Colorbar height is zero, cannot resample.")
        new_width = int(cb_width * (target_height / cb_height))
        resized_cb = colorbar_img.resize(
            (new_width, target_height), Image.Resampling.LANCZOS
        )
        resized_cb.save(colorbar_path, format="PNG")


def make_transparent_png(width: int, height: int) -> bytes:
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def placeholder_from_png(path: Path) -> bytes:
    with Image.open(path) as img:
        width, height = img.size
    return make_transparent_png(width, height)


class FilterButtonGrid:
    FILTER_ATTR = {
        DatabaseSampleColumn.ANTENNA_TYPE: "antenna_type",
        DatabaseSampleColumn.FREQUENCY: "frequency",
        DatabaseSampleColumn.DISTANCE: "distance",
        DatabaseSampleColumn.MASS: "mass",
    }

    def __init__(
        self,
        catalog: DatabaseSampleCatalog,
        on_change: Callable[[], None] | None = None,
    ):
        self.catalog = catalog
        self.on_change = on_change
        self.filter_options = DatabaseSampleFilters()
        self.filtered_df = self.catalog.filter_dataframe(self.filter_options)
        self.button_groups: dict[DatabaseSampleColumn, list[widgets.ToggleButton]] = {}

    def create_radio_button_grid(self) -> widgets.HBox:
        all_button_columns: list[widgets.VBox] = []
        for (
            column_name,
            unique_values,
        ) in self.catalog.unique_entries_in_columns().items():
            column_enum = DatabaseSampleColumn(column_name)
            if column_enum == DatabaseSampleColumn.FILE_PATH:
                continue
            buttons: list[widgets.ToggleButton] = []
            for value in unique_values:
                btn = widgets.ToggleButton(
                    description=str(value),
                    disabled=False,
                    layout=widgets.Layout(width="100%"),
                )
                btn.observe(self._make_handler(btn, column_enum), "value")
                buttons.append(btn)
            self.button_groups[column_enum] = buttons
            all_button_columns.append(
                widgets.VBox(
                    [widgets.HTML(f"<b>{column_name}</b>"), *buttons],
                    layout=widgets.Layout(
                        flex="1 1 0%",
                        min_width="0",
                        overflow="hidden",
                    ),
                )
            )
        return widgets.HBox(all_button_columns, layout=widgets.Layout(width="100%"))

    def _coerce_value(
        self, column_enum: DatabaseSampleColumn, raw_value: str
    ) -> str | float:
        if column_enum == DatabaseSampleColumn.ANTENNA_TYPE:
            return raw_value
        return float(raw_value)

    def _make_handler(
        self, button: widgets.ToggleButton, column_enum: DatabaseSampleColumn
    ) -> Callable[[Bunch], None]:
        def handler(change: Bunch) -> None:
            if change["name"] != "value":
                return
            column_attr = self.FILTER_ATTR[column_enum]
            value = self._coerce_value(column_enum, change.owner.description)
            group_buttons = self.button_groups[column_enum]
            if change["new"]:
                for sibling in group_buttons:
                    if sibling is not button:
                        sibling.value = False
                setattr(
                    self.filter_options,
                    column_attr,
                    DatabaseSampleFilterOption(column_name=column_enum, value=value),
                )
            else:
                current_filter = getattr(self.filter_options, column_attr)
                if current_filter and current_filter.value == value:
                    setattr(self.filter_options, column_attr, None)
            self.filtered_df = self.catalog.filter_dataframe(self.filter_options)
            self.update_button_states(self.filtered_df)
            if self.on_change is not None:
                self.on_change()

        return handler

    def update_button_states(self, filtered_df: pd.DataFrame) -> None:
        for column_enum, buttons in self.button_groups.items():
            valid_values = set(filtered_df[column_enum.value].astype(str))
            for button in buttons:
                button.disabled = button.description not in valid_values

    def apply_filters(self, filters: DatabaseSampleFilters) -> None:
        self.filter_options = filters.model_copy(deep=True)
        self.filtered_df = self.catalog.filter_dataframe(self.filter_options)
        for column_enum, attr_name in self.FILTER_ATTR.items():
            selected = getattr(self.filter_options, attr_name)
            selected_description = None if selected is None else str(selected.value)
            for button in self.button_groups.get(column_enum, []):
                button.unobserve_all("value")
                button.value = button.description == selected_description
                button.observe(self._make_handler(button, column_enum), "value")
        self.update_button_states(self.filtered_df)
        if self.on_change is not None:
            self.on_change()

    @property
    def selected_reference_path(self) -> Path | None:
        if len(self.filtered_df) != 1:
            return None
        return Path(self.filtered_df[DatabaseSampleColumn.FILE_PATH.value].iloc[0])


class SarGammaComparisonUI:
    def __init__(self, paths: WorkspacePaths | None = None):
        ensure_notebook_prerequisites()
        self.paths = paths or default_workspace_paths()
        self.paths.ensure_runtime_dirs()

        self.logging_window = OutputWidgetHandler()
        self.logging_window.setFormatter(
            logging.Formatter("%(asctime)s  - [%(levelname)s] %(message)s")
        )
        # Attach handler to the package logger so runner.py logs reach the widget too
        _frontend_logger = logging.getLogger("sar_pattern_validation.voila_frontend")
        _frontend_logger.handlers.clear()
        _frontend_logger.addHandler(self.logging_window)
        _frontend_logger.setLevel(logging.INFO)
        _frontend_logger.propagate = False

        self.logger = logging.getLogger(__name__)

        self.catalog = DatabaseSampleCatalog.scan(self.paths.database_path)
        self.runner = SarPatternValidationRunner(self.paths)
        self.workflow_results: WorkflowResultPayload | None = None
        self._progress_thread = None
        self._stop_event = None

        self.radio_button_grid = FilterButtonGrid(self.catalog, self._on_filter_change)
        display(self.create_ui())
        self.restore_state()

    def _build_state(self) -> UiState:
        return UiState(
            measured_file_name=self.uploaded_file_name_label.value,
            power_level=float(self.power_level.value),
            active_filters=self.radio_button_grid.filter_options,
            last_result=self.workflow_results,
        )

    def _persist_state(self) -> None:
        save_ui_state(self.paths, self._build_state())

    def _on_filter_change(self) -> None:
        self._refresh_run_button_state()
        self._persist_state()

    def _refresh_run_button_state(self) -> None:
        self.run_button.disabled = not (
            self.paths.measured_file_path.exists()
            and self.radio_button_grid.selected_reference_path is not None
        )

    def _start_progress_updater(self) -> None:
        import threading

        self._stop_event = threading.Event()
        with self.progress_output:
            self.progress_output.clear_output()
            display(self.progress_bar)

        def update_progress() -> None:
            duration = 240
            interval = 0.1
            steps = int(duration / interval)
            for index in range(steps):
                if self._stop_event is not None and self._stop_event.is_set():
                    break
                self.progress_bar.value = min(0.9, index / steps * 0.9)
                time.sleep(interval)

        self.progress_bar.value = 0.0
        self.progress_bar.bar_style = "info"
        self.progress_bar.style = {"bar_color": GuiColors.PRIMARY.value}
        self._progress_thread = threading.Thread(target=update_progress, daemon=True)
        self._progress_thread.start()

    def _stop_progress_updater(self, *, completed: bool) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._progress_thread and self._progress_thread.is_alive():
            self._progress_thread.join(timeout=1.0)
        if completed:
            for value in range(int(self.progress_bar.value * 100), 101, 5):
                self.progress_bar.value = value / 100
                time.sleep(0.05)
            self.progress_bar.value = self.progress_bar.max
        self.progress_bar.style = {
            "bar_color": (
                GuiColors.PRIMARY.value if completed else GuiColors.FAIL.value
            )
        }
        self.progress_output.clear_output()

    def handle_button_click(self, button: widgets.Button) -> None:
        button.style = {
            "button_color": GuiColors.LOADING.value,
            "text_color": GuiColors.TEXT_PRIMARY.value,
        }
        button.disabled = True
        self._start_progress_updater()
        try:
            reference_path = self.radio_button_grid.selected_reference_path
            if reference_path is None:
                raise RuntimeError("Select exactly one reference configuration.")
            self.workflow_results = self.runner.run_workflow(
                reference_file_path=reference_path,
                power_level_dbm=float(self.power_level.value),
            )
            self._stop_progress_updater(completed=True)
            self._persist_state()
            self.update_images()
            self._update_analytical_results(self.workflow_results)
            self.logger.info("SAR Pattern Validation done.")
        except Exception as error:  # noqa: BLE001
            self._stop_progress_updater(completed=False)
            button.style = {
                "button_color": GuiColors.FAIL.value,
                "text_color": GuiColors.TEXT_PRIMARY.value,
            }
            self.logger.warning(error)
        finally:
            button.style = {
                "button_color": GuiColors.PRIMARY.value,
                "text_color": GuiColors.TEXT_PRIMARY.value,
            }
            self._refresh_run_button_state()

    def _update_analytical_results(self, results: WorkflowResultPayload) -> None:
        passed = results.passed_pixel_count == results.evaluated_pixel_count
        self.result_indicator_button.description = "Pass" if passed else "Fail"
        self.result_indicator_button.style = {
            "button_color": (
                GuiColors.PRIMARY.value if passed else GuiColors.FAIL.value
            ),
            "text_color": GuiColors.TEXT_PRIMARY.value,
        }
        self.pass_rate_label.value = (
            f"<b>Pass rate = {results.pass_rate_percent:.1f}%</b>"
        )
        values = [
            f"{results.reference_pssar:.2f}",
            f"{results.measured_pssar:.2f}",
            f"{100 * results.scaling_error:.2f}",
        ]
        self.result_table.value = f"""
        <table style="border-collapse: collapse; width: 100%; font-family: Arial;">
            <thead>
                <tr style="background-color: #f2f2f2;">
                    <th style="border: 1px solid #ddd; padding: 8px;"></th>
                    {"".join(f'<th style="border: 1px solid #ddd; padding: 8px;">{col.value}</th>' for col in ResultTableColumn)}
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;"><b>{ResultTableRow.SSAR.value}</b></td>
                    {"".join(f'<td style="border: 1px solid #ddd; padding: 8px;">{value}</td>' for value in values)}
                </tr>
            </tbody>
        </table>
        """

    def _on_file_upload_change(self, change: Bunch) -> None:
        value = change["new"]
        if not value:
            self.uploaded_file_name_label.value = ""
            self._persist_state()
            self._refresh_run_button_state()
            return

        file_info = value[0]
        self.uploaded_file_name_label.value = str(file_info["name"])
        self.paths.measured_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.paths.measured_file_path.write_bytes(file_info["content"])
        self._persist_state()
        self._refresh_run_button_state()

    def restore_state(self) -> None:
        state = load_or_migrate_ui_state(self.paths)
        if state is None:
            self.update_images(no_data=True)
            self._refresh_run_button_state()
            return

        self.uploaded_file_name_label.value = state.measured_file_name
        self.power_level.value = state.power_level
        self.radio_button_grid.apply_filters(state.active_filters)
        self.workflow_results = state.last_result
        if self.workflow_results is not None and self._restore_outputs_available():
            self.update_images(no_data=False)
            self._update_analytical_results(self.workflow_results)
            self.logger.info("Session state restored successfully")
        else:
            self.update_images(no_data=True)
        self._refresh_run_button_state()

    def _restore_outputs_available(self) -> bool:
        required_files = [
            self.paths.measured_file_path,
            self.paths.reference_image_path,
            self.paths.measured_image_path,
            self.paths.aligned_means_path,
            self.paths.registered_image_path,
            self.paths.gamma_comparison_path,
            self.paths.gamma_comparison_failures_path,
        ]
        return all(path.exists() for path in required_files)

    def update_images(self, *, no_data: bool = False) -> None:
        if no_data:
            image_files: list[Path] = [self.paths.no_data_image] * 8
        else:
            resample_colorbar_to_match_plot_inplace(
                self.paths.aligned_means_colorbar_path,
                self.paths.aligned_means_path,
            )
            resample_colorbar_to_match_plot_inplace(
                self.paths.gamma_comparison_colorbar_path,
                self.paths.gamma_comparison_path,
            )
            image_files = [
                self.paths.reference_image_path,
                self.paths.measured_image_path,
                self.paths.aligned_means_path,
                self.paths.aligned_means_colorbar_path,
                self.paths.registered_image_path,
                self.paths.gamma_comparison_path,
                self.paths.gamma_comparison_colorbar_path,
                self.paths.gamma_comparison_failures_path,
            ]

        widgets_list = [
            self.image_top_left,
            self.image_top_middle,
            self.image_top_right,
            self.colorbar_top,
            self.image_bottom_left,
            self.image_bottom_middle,
            self.colorbar_bottom,
            self.image_bottom_right,
        ]
        for img_widget, path in zip(widgets_list, image_files, strict=True):
            if no_data:
                img_widget.value = TRANSPARENT_PNG
            else:
                try:
                    img_widget.value = path.read_bytes()
                except FileNotFoundError:
                    img_widget.value = TRANSPARENT_PNG

        placeholder_png = (
            TRANSPARENT_PNG
            if no_data
            else placeholder_from_png(self.paths.aligned_means_colorbar_path)
        )
        for placeholder in (
            self.placeholder_1,
            self.placeholder_2,
            self.placeholder_3,
            self.placeholder_4,
        ):
            placeholder.value = placeholder_png

    def create_ui(
        self,
        *,
        left_ratio: float = 0.3,
        right_ratio: float = 0.7,
        side_gap: str = "100px",
    ) -> widgets.VBox:
        def row(children, gap: str = "10px", align: str = "center") -> widgets.HBox:
            return widgets.HBox(
                children=children,
                layout=widgets.Layout(
                    gap=gap,
                    width="100%",
                    align_items=align,
                    overflow="hidden",
                ),
            )

        def col(children, gap: str = "10px") -> widgets.VBox:
            return widgets.VBox(
                children=children,
                layout=widgets.Layout(
                    gap=gap,
                    width="100%",
                    align_items="stretch",
                    overflow="hidden",
                ),
            )

        def flex_item(widget, flex="1", min_width="0"):
            widget.layout = widgets.Layout(
                flex=flex, min_width=min_width, overflow="hidden"
            )
            return widget

        self.upload_1 = widgets.FileUpload(
            accept=".csv",
            multiple=False,
            description="Measured CSV",
            style={
                "button_color": GuiColors.PRIMARY.value,
                "text_color": GuiColors.TEXT_PRIMARY.value,
            },
        )
        self.upload_1.observe(self._on_file_upload_change, names="value")
        self.power_level = widgets.BoundedFloatText(
            value=23.0,
            min=-10,
            max=50,
            step=0.1,
            description="power level [dBm]:",
            style={"description_width": "initial"},
        )
        self.power_level.observe(lambda change: self._persist_state(), names="value")
        self.uploaded_file_name_label = widgets.Label(value="")

        tooltip = widgets.HTML(
            value="""
            <div style="
                background-color: #E8F6FD;
                color: #005A8C;
                border-left: 4px solid #0090D0;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 14px;
            ">
                <b>ℹ️ Note:</b> The uploaded <code>.csv</code> files must be smaller than <b>10 MB</b>.
            </div>
            """
        )

        left_setup_section = col(
            [
                row(
                    [
                        flex_item(self.upload_1, "1"),
                        flex_item(self.power_level, "1", "150px"),
                    ]
                ),
                self.uploaded_file_name_label,
                tooltip,
                self.radio_button_grid.create_radio_button_grid(),
            ]
        )
        left_setup_section.layout.flex = str(left_ratio)

        self.run_button = widgets.Button(
            description="Compare Patterns",
            style={
                "button_color": GuiColors.PRIMARY.value,
                "text_color": GuiColors.TEXT_PRIMARY.value,
            },
            disabled=True,
        )
        self.run_button.on_click(self.handle_button_click)
        self.result_indicator_button = widgets.Button(
            description="",
            style={
                "button_color": GuiColors.WHITE.value,
                "text_color": GuiColors.TEXT_PRIMARY.value,
            },
        )
        self.pass_rate_label = widgets.HTML(value="")
        self.result_table = widgets.HTML(value="")
        result_info_group = row(
            [
                flex_item(self.result_indicator_button, "0 0 80px"),
                flex_item(self.pass_rate_label, "0 0 auto"),
                flex_item(self.result_table, "2"),
            ],
            gap="10px",
        )
        results_top_section = row(
            [flex_item(self.run_button, "0 0 auto"), result_info_group],
            gap="40px",
        )

        self.progress_bar = widgets.FloatProgress(
            value=0.0,
            min=0.0,
            max=1.0,
            description="Progress:",
            bar_style="info",
            style={"bar_color": GuiColors.PRIMARY.value},
            layout=widgets.Layout(width="95%"),
        )
        self.progress_output = widgets.Output(layout=widgets.Layout(width="100%"))
        progress_bar_container = row(
            [self.progress_output], gap="0px", align="flex-start"
        )

        def wrap_image(img: widgets.Image, flex: str = "1") -> widgets.Box:
            return widgets.Box(
                [img],
                layout=widgets.Layout(
                    flex=flex,
                    height="100%",
                    overflow="hidden",
                    align_items="stretch",
                    justify_content="center",
                ),
            )

        def create_main_image() -> tuple[widgets.Image, widgets.Box]:
            image = widgets.Image(
                format="png",
                layout=widgets.Layout(
                    width="100%", height="100%", object_fit="contain"
                ),
            )
            return image, wrap_image(image, flex="4")

        def create_colorbar_image() -> tuple[widgets.Image, widgets.Box]:
            image = widgets.Image(
                format="png",
                layout=widgets.Layout(
                    width="100%", height="100%", object_fit="contain"
                ),
            )
            return image, wrap_image(image, flex="1")

        self.image_top_left, box_top_left = create_main_image()
        self.image_top_middle, box_top_middle = create_main_image()
        self.image_top_right, box_top_right = create_main_image()
        self.image_bottom_left, box_bottom_left = create_main_image()
        self.image_bottom_middle, box_bottom_middle = create_main_image()
        self.image_bottom_right, box_bottom_right = create_main_image()
        self.colorbar_top, box_cb_top = create_colorbar_image()
        self.colorbar_bottom, box_cb_bottom = create_colorbar_image()
        self.placeholder_1, box_ph_1 = create_colorbar_image()
        self.placeholder_2, box_ph_2 = create_colorbar_image()
        self.placeholder_3, box_ph_3 = create_colorbar_image()
        self.placeholder_4, box_ph_4 = create_colorbar_image()
        self.update_images(no_data=True)

        row_layout = widgets.Layout(
            width="100%", align_items="stretch", justify_content="space-between"
        )
        top_row = widgets.HBox(
            [
                widgets.HBox(
                    [box_top_left, box_ph_1],
                    layout=widgets.Layout(width="33%", align_items="stretch"),
                ),
                widgets.HBox(
                    [box_top_middle, box_ph_2],
                    layout=widgets.Layout(width="33%", align_items="stretch"),
                ),
                widgets.HBox(
                    [box_top_right, box_cb_top],
                    layout=widgets.Layout(width="33%", align_items="stretch"),
                ),
            ],
            layout=row_layout,
        )
        bottom_row = widgets.HBox(
            [
                widgets.HBox(
                    [box_bottom_left, box_ph_3],
                    layout=widgets.Layout(width="33%", align_items="stretch"),
                ),
                widgets.HBox(
                    [box_bottom_middle, box_cb_bottom],
                    layout=widgets.Layout(width="33%", align_items="stretch"),
                ),
                widgets.HBox(
                    [box_bottom_right, box_ph_4],
                    layout=widgets.Layout(width="33%", align_items="stretch"),
                ),
            ],
            layout=row_layout,
        )
        images_section = widgets.Box(
            [widgets.VBox([top_row, bottom_row], layout=widgets.Layout(width="100%"))],
            layout=widgets.Layout(max_height="600px", overflow_y="auto", padding="5px"),
        )

        right_results_section = col(
            [results_top_section, progress_bar_container, images_section]
        )
        right_results_section.layout.flex = str(right_ratio)
        main_gui_section = widgets.HBox(
            [left_setup_section, right_results_section],
            layout=widgets.Layout(gap=side_gap, width="100%", align_items="flex-start"),
        )
        return widgets.VBox(
            [main_gui_section, self.logging_window.show_logs()],
            layout=widgets.Layout(gap="10px", width="100%", align_items="flex-start"),
        )


def bootstrap_voila_ui(paths: WorkspacePaths | None = None) -> SarGammaComparisonUI:
    return SarGammaComparisonUI(paths=paths)

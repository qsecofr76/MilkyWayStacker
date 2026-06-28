import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from ui.canvas import MaskingCanvas
from core.stacker import stack_images, load_image, apply_gamma
from core.aligner import check_features, get_debug_matches_image, draw_constellations, get_debug_stars_image

# Set CustomTkinter theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class MilkyWayStackerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Milky Way Stacker")
        self.geometry("1200x950")
        
        # State variables
        self.image_paths = []
        self.original_reference_img = None
        self.reference_img = None
        self.output_img = None
        self.show_constellations_var = ctk.BooleanVar(value=False)

        self._create_widgets()
        
        # Register canvas brush size scroll wheel callback
        self.canvas.on_brush_size_changed = self._update_brush_slider

    def _create_widgets(self):
        # Configure grid layout: 2 columns, 2 rows
        self.grid_columnconfigure(0, weight=0)  # Left panel (fixed width)
        self.grid_columnconfigure(1, weight=1)  # Main canvas (resizable)
        self.grid_rowconfigure(0, weight=1)     # Main area
        self.grid_rowconfigure(1, weight=0)     # Bottom status area

        # --- Sidebar (Left Panel, now Scrollable) ---
        self.sidebar = ctk.CTkScrollableFrame(self, width=340, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=10, pady=10)

        # Load and display Association Logo (logo.png)
        logo_path = "logo.png"
        if os.path.exists(logo_path):
            try:
                logo_pil = Image.open(logo_path)
                orig_w, orig_h = logo_pil.size
                ratio = min(200 / orig_w, 70 / orig_h)
                new_w = int(orig_w * ratio)
                new_h = int(orig_h * ratio)
                
                self.logo_image = ctk.CTkImage(light_image=logo_pil, dark_image=logo_pil, size=(new_w, new_h))
                self.logo_label = ctk.CTkLabel(self.sidebar, image=self.logo_image, text="")
                self.logo_label.pack(pady=(15, 0))
            except Exception as e:
                print(f"Failed to load logo.png: {e}")

        # App Title
        self.title_label = ctk.CTkLabel(self.sidebar, text="MilkyWay Stacker", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(pady=(10, 10))

        # Files Section
        self.load_btn = ctk.CTkButton(self.sidebar, text="Load Images", command=self.load_images)
        self.load_btn.pack(fill="x", padx=20, pady=5)

        self.files_listbox = tk.Listbox(self.sidebar, bg="#2d2d2d", fg="white", borderwidth=0, selectbackground="#1f538d", height=6)
        self.files_listbox.pack(fill="x", padx=20, pady=5)

        # Paint Mask Tools
        self.brush_label = ctk.CTkLabel(self.sidebar, text="Masking Brush Controls", font=ctk.CTkFont(size=14, weight="bold"))
        self.brush_label.pack(pady=(10, 2))

        self.brush_size_label = ctk.CTkLabel(self.sidebar, text="Brush Size: 30")
        self.brush_size_label.pack()
        self.brush_size_slider = ctk.CTkSlider(self.sidebar, from_=5, to=100, number_of_steps=95, command=self.change_brush_size)
        self.brush_size_slider.set(30)
        self.brush_size_slider.pack(fill="x", padx=20, pady=2)

        # Mask Management: Save, Load, Clear
        self.mask_btns_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.mask_btns_frame.pack(fill="x", padx=20, pady=5)

        self.save_mask_btn = ctk.CTkButton(self.mask_btns_frame, text="Save Mask", command=self.save_mask)
        self.save_mask_btn.grid(row=0, column=0, padx=2, pady=2, sticky="ew")

        self.load_mask_btn = ctk.CTkButton(self.mask_btns_frame, text="Load Mask", command=self.load_mask)
        self.load_mask_btn.grid(row=0, column=1, padx=2, pady=2, sticky="ew")

        self.select_all_sky_btn = ctk.CTkButton(self.mask_btns_frame, text="Select All Sky", command=self.select_all_sky)
        self.select_all_sky_btn.grid(row=1, column=0, padx=2, pady=2, sticky="ew")

        self.clear_mask_btn = ctk.CTkButton(self.mask_btns_frame, text="Clear Mask", fg_color="transparent", border_width=1, command=self.clear_mask)
        self.clear_mask_btn.grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        
        self.mask_btns_frame.grid_columnconfigure(0, weight=1)
        self.mask_btns_frame.grid_columnconfigure(1, weight=1)

        # Feature Detection Options
        self.features_label = ctk.CTkLabel(self.sidebar, text="Feature Detection Settings", font=ctk.CTkFont(size=14, weight="bold"))
        self.features_label.pack(pady=(15, 2))

        # Contrast Threshold (Sensitivity)
        self.contrast_label = ctk.CTkLabel(self.sidebar, text="Sensitivity (Contrast): 0.04\n(lower detects more dim stars/details)")
        self.contrast_label.pack()
        self.contrast_slider = ctk.CTkSlider(self.sidebar, from_=0.005, to=0.15, number_of_steps=100, command=self.change_contrast_threshold)
        self.contrast_slider.set(0.04)
        self.contrast_slider.pack(fill="x", padx=20, pady=2)

        # Sigma (Gaussian blur)
        self.sigma_label = ctk.CTkLabel(self.sidebar, text="Star Gaussian Blur (Sigma): 1.6")
        self.sigma_label.pack()
        self.sigma_slider = ctk.CTkSlider(self.sidebar, from_=0.5, to=3.0, number_of_steps=25, command=self.change_sigma)
        self.sigma_slider.set(1.6)
        self.sigma_slider.pack(fill="x", padx=20, pady=2)

        # Gamma Correction
        self.gamma_label = ctk.CTkLabel(self.sidebar, text="Gamma Correction: 1.0\n(lower brightens, higher darkens)")
        self.gamma_label.pack()
        self.gamma_slider = ctk.CTkSlider(self.sidebar, from_=0.1, to=3.0, number_of_steps=29, command=self.change_gamma)
        self.gamma_slider.set(1.0)
        self.gamma_slider.pack(fill="x", padx=20, pady=2)

        # Stacking & Alignment Geometry Parameters
        self.params_label = ctk.CTkLabel(self.sidebar, text="Stacking & Alignment Geometry", font=ctk.CTkFont(size=14, weight="bold"))
        self.params_label.pack(pady=(15, 2))

        # Transform type
        self.transform_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.transform_frame.pack(fill="x", padx=20, pady=2)
        self.transform_label = ctk.CTkLabel(self.transform_frame, text="Align Method:")
        self.transform_label.pack(side="left")
        self.transform_menu = ctk.CTkComboBox(self.transform_frame, values=["affine", "homography"], width=120)
        self.transform_menu.set("affine")
        self.transform_menu.pack(side="right")

        self.stack_mode_menu_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.stack_mode_menu_frame.pack(fill="x", padx=20, pady=2)
        self.stack_mode_label = ctk.CTkLabel(self.stack_mode_menu_frame, text="Stack Mode:")
        self.stack_mode_label.pack(side="left")
        self.stack_mode_menu = ctk.CTkComboBox(self.stack_mode_menu_frame, values=["average", "median"], width=120)
        self.stack_mode_menu.set("average")
        self.stack_mode_menu.pack(side="right")

        # Freeze Ground checkbox
        self.freeze_ground_var = ctk.BooleanVar(value=False)
        self.freeze_ground_cb = ctk.CTkCheckBox(
            self.sidebar, text="Freeze Ground (Reference Frame)", 
            variable=self.freeze_ground_var
        )
        self.freeze_ground_cb.pack(anchor="w", padx=20, pady=5)

        self.feather_label = ctk.CTkLabel(self.sidebar, text="Feathering Radius: 10px")
        self.feather_label.pack()
        self.feather_slider = ctk.CTkSlider(self.sidebar, from_=0, to=250, number_of_steps=250, command=self.change_feather_radius)
        self.feather_slider.set(10)
        self.feather_slider.pack(fill="x", padx=20, pady=2)

        # Pre-launch and Process / Save Buttons
        self.checks_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.checks_frame.pack(fill="x", padx=20, pady=(15, 2))
        
        self.check_btn = ctk.CTkButton(self.checks_frame, text="Pre-Launch", width=120, fg_color="#cf830e", hover_color="#ad6c0a", command=self.run_prelaunch_check)
        self.check_btn.grid(row=0, column=0, padx=2, sticky="ew")

        self.debug_btn = ctk.CTkButton(self.checks_frame, text="Debug Matches", width=120, fg_color="#7b2cbf", hover_color="#5a189a", command=self.run_debug_matches)
        self.debug_btn.grid(row=0, column=1, padx=2, sticky="ew")
        
        self.checks_frame.grid_columnconfigure(0, weight=1)
        self.checks_frame.grid_columnconfigure(1, weight=1)

        self.stack_btn = ctk.CTkButton(self.sidebar, text="Stack Images", fg_color="#2b8c44", hover_color="#1d662e", command=self.start_stacking)
        self.stack_btn.pack(fill="x", padx=20, pady=5)

        self.constellation_cb = ctk.CTkCheckBox(
            self.sidebar, text="Show Constellations", 
            state="disabled", variable=self.show_constellations_var, 
            command=self.toggle_constellations
        )
        self.constellation_cb.pack(fill="x", padx=20, pady=5)

        self.save_btn = ctk.CTkButton(self.sidebar, text="Save Result", fg_color="#1f538d", state="disabled", command=self.save_result)
        self.save_btn.pack(fill="x", padx=20, pady=2)

        # --- Main Image Canvas ---
        self.canvas = MaskingCanvas(self, bg="#1a1a1a", highlightthickness=0)
        self.canvas.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=(10, 5))

        # --- Bottom Status Bar ---
        self.status_frame = ctk.CTkFrame(self, height=40, corner_radius=0)
        self.status_frame.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(0, 10))

        self.status_label = ctk.CTkLabel(self.status_frame, text="Ready. Load images to begin.", anchor="w")
        self.status_label.pack(side="left", padx=10, fill="x", expand=True)

        self.progress_bar = ctk.CTkProgressBar(self.status_frame, width=200)
        self.progress_bar.pack(side="right", padx=10, pady=10)
        self.progress_bar.set(0)

    def load_images(self):
        files = filedialog.askopenfilenames(
            title="Select Images for Stacking",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.dng *.fit *.fits")]
        )
        if not files:
            return

        self.image_paths = sorted(list(files))
        
        # Update Listbox
        self.files_listbox.delete(0, tk.END)
        for path in self.image_paths:
            self.files_listbox.insert(tk.END, os.path.basename(path))

        # Load reference image (middle frame of sequence to minimize overall distortion)
        ref_idx = len(self.image_paths) // 2
        ref_path = self.image_paths[ref_idx]
        self.status_label.configure(text=f"Loading reference frame (middle): {os.path.basename(ref_path)}...")
        self.update_idletasks()

        self.canvas.show_mask = True
        self.original_reference_img = load_image(ref_path)
        if self.original_reference_img is not None:
            self.reference_img = self.original_reference_img.copy()
            self.canvas.set_image(self.reference_img)
            self.status_label.configure(text=f"Loaded {len(self.image_paths)} images. Paint the sky mask on the reference frame, then click 'Stack Images'.")
            self.constellation_cb.configure(state="normal")
            self.save_btn.configure(state="disabled")
            self.update_display_image()
        else:
            self.status_label.configure(text="Failed to load reference image.")

    def _update_brush_slider(self, val):
        self.brush_size_slider.set(val)
        self.brush_size_label.configure(text=f"Brush Size: {val}")

    def change_brush_size(self, val):
        size = int(val)
        self.brush_size_label.configure(text=f"Brush Size: {size}")
        self.canvas.set_brush_size(size)

    def change_contrast_threshold(self, val):
        self.contrast_label.configure(text=f"Sensitivity (Contrast): {val:.3f}")

    def change_sigma(self, val):
        self.sigma_label.configure(text=f"Star Gaussian Blur (Sigma): {val:.1f}")

    def change_gamma(self, val):
        gamma = float(val)
        self.gamma_label.configure(text=f"Gamma Correction: {gamma:.1f}\n(lower brightens, higher darkens)")
        if self.original_reference_img is not None:
            self.reference_img = apply_gamma(self.original_reference_img, gamma)
            self.canvas.set_image(self.reference_img, reset_mask=False)

    def change_feather_radius(self, val):
        radius = int(val)
        self.feather_label.configure(text=f"Feathering Radius: {radius}px")

    def clear_mask(self):
        self.canvas.clear_mask()

    def select_all_sky(self):
        self.canvas.fill_mask_sky()

    def save_mask(self):
        if self.reference_img is None:
            messagebox.showwarning("No Image", "Please load an image first before saving a mask.")
            return
        file_path = filedialog.asksaveasfilename(
            title="Save Sky Mask",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")]
        )
        if file_path:
            if self.canvas.save_mask_to_file(file_path):
                self.status_label.configure(text=f"Saved sky mask to: {os.path.basename(file_path)}")
                messagebox.showinfo("Saved", "Mask saved successfully.")

    def load_mask(self):
        if self.reference_img is None:
            messagebox.showwarning("No Image", "Please load an image first before loading a mask.")
            return
        file_path = filedialog.askopenfilename(
            title="Load Sky Mask",
            filetypes=[("PNG Image", "*.png")]
        )
        if file_path:
            if self.canvas.load_mask_from_file(file_path):
                self.status_label.configure(text=f"Loaded sky mask: {os.path.basename(file_path)}")
                messagebox.showinfo("Loaded", "Mask loaded successfully.")

    def run_prelaunch_check(self):
        if self.reference_img is None:
            messagebox.showwarning("No Image", "Please load images first.")
            return

        mask = self.canvas.get_mask()
        contrast = float(self.contrast_slider.get())
        sigma = float(self.sigma_slider.get())

        sky_count, ground_count = check_features(
            self.reference_img, mask,
            contrast_threshold=contrast,
            edge_threshold=10.0,
            sigma=sigma
        )

        msg = (
            f"Pre-Launch Alignment Check:\n\n"
            f"Stars detected in Sky region: {sky_count}\n"
            f"Alignment points in Landscape region: {ground_count}\n\n"
        )
        if sky_count < 10:
            msg += "⚠️ Warning: Too few stars in the sky region. Lower 'Sensitivity (Contrast)' to detect more dim stars.\n"
        else:
            msg += "✅ Sky region looks ready for star alignment.\n"

        has_ground = np.any(mask == 0)
        if has_ground:
            if self.freeze_ground_var.get():
                msg += "ℹ️ Note: Ground alignment is skipped (Freeze Ground is enabled).\n"
            elif ground_count < 10:
                msg += "⚠️ Warning: Too few alignment points in landscape region. Lower 'Sensitivity (Contrast)' to capture more ground details.\n"
            else:
                msg += "✅ Landscape region looks ready for foreground alignment.\n"
        else:
            msg += "ℹ️ Note: Ground alignment is skipped (entire image masked as sky).\n"

        messagebox.showinfo("Pre-Launch Check", msg)

    def run_debug_matches(self):
        if len(self.image_paths) < 2:
            messagebox.showwarning("Incomplete sequence", "Please load at least 2 images to compare and debug matching.")
            return
        
        # Load second image to compare and apply current gamma
        gamma = float(self.gamma_slider.get())
        second_img = load_image(self.image_paths[1])
        if second_img is None:
            messagebox.showerror("Error", f"Failed to load second image: {self.image_paths[1]}")
            return
        second_img = apply_gamma(second_img, gamma)

        mask = self.canvas.get_mask()
        contrast = float(self.contrast_slider.get())
        sigma = float(self.sigma_slider.get())

        # Ask user which part to debug
        choice_win = ctk.CTkToplevel(self)
        choice_win.title("Debug Mode Selection")
        choice_win.geometry("380x220")
        choice_win.resizable(False, False)
        choice_win.attributes("-topmost", True)
        choice_win.focus_force()
        
        # Center choice win
        choice_win.update_idletasks()
        w = choice_win.winfo_width()
        h = choice_win.winfo_height()
        x = (choice_win.winfo_screenwidth() // 2) - (w // 2)
        y = (choice_win.winfo_screenheight() // 2) - (h // 2)
        choice_win.geometry(f"+{x}+{y}")
        
        label = ctk.CTkLabel(choice_win, text="Select the debug tool to execute:", font=ctk.CTkFont(size=14, weight="bold"))
        label.pack(pady=15)

        def debug_sky():
            choice_win.destroy()
            self._show_debug_window(self.reference_img, second_img, mask, True, contrast, sigma)

        def debug_ground():
            if self.freeze_ground_var.get():
                messagebox.showwarning("Ground Frozen", "Ground alignment debug is not available because 'Freeze Ground' is active.")
                choice_win.destroy()
                return
            choice_win.destroy()
            # Load first and last images to compare the maximum drift and apply current gamma
            first_img = load_image(self.image_paths[0])
            last_img = load_image(self.image_paths[-1])
            if first_img is None or last_img is None:
                messagebox.showerror("Error", "Failed to load first or last image in sequence.")
                return
            first_img = apply_gamma(first_img, gamma)
            last_img = apply_gamma(last_img, gamma)
            self._show_debug_window(first_img, last_img, mask, False, contrast, sigma)

        def debug_detected_stars():
            choice_win.destroy()
            self._show_stars_debug_window(mask)

        btn_stars_detect = ctk.CTkButton(choice_win, text="Show Recognizable Stars (Sky)", fg_color="#00a896", hover_color="#028090", command=debug_detected_stars, width=300)
        btn_stars_detect.pack(pady=5)

        btn_sky = ctk.CTkButton(choice_win, text="Sky Star Alignment Matches", command=debug_sky, width=300)
        btn_sky.pack(pady=5)

        btn_ground = ctk.CTkButton(choice_win, text="Landscape Alignment Matches", command=debug_ground, width=300)
        btn_ground.pack(pady=5)

    def _show_stars_debug_window(self, mask):
        self.status_label.configure(text="Generating recognizable stars debug image...")
        self.update_idletasks()

        contrast = float(self.contrast_slider.get())
        sigma = float(self.sigma_slider.get())
        debug_img = get_debug_stars_image(self.reference_img, mask, contrast_threshold=contrast, sigma=sigma)

        title = "Recognizable Stars - Sky Region"
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 1000, 700)
        cv2.imshow(title, debug_img)
        self.status_label.configure(text="Opened recognizable stars window. Press any key in that window to close it.")

    def _show_debug_window(self, ref_img, target_img, mask, align_sky, contrast, sigma):
        self.status_label.configure(text="Generating diagnostic matches image...")
        self.update_idletasks()

        # Run diagnostic image generation
        debug_img = get_debug_matches_image(
            ref_img, target_img, mask, align_sky,
            contrast_threshold=contrast, edge_threshold=10.0, sigma=sigma
        )

        title = "Diagnostic Matches - Sky (Stars)" if align_sky else "Diagnostic Matches - Landscape"
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 1200, 700)
        cv2.imshow(title, debug_img)
        self.status_label.configure(text="Opened matches visualization window. Press any key in that window to close it.")

    def start_stacking(self):
        if not self.image_paths:
            messagebox.showwarning("No Images", "Please load images first.")
            return

        # Start stacking in background thread to prevent UI freezing
        self.stack_btn.configure(state="disabled")
        self.load_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        self.constellation_cb.configure(state="disabled")
        
        mode = self.stack_mode_menu.get()
        feather = int(self.feather_slider.get())
        mask = self.canvas.get_mask()
        contrast = float(self.contrast_slider.get())
        sigma = float(self.sigma_slider.get())
        gamma = float(self.gamma_slider.get())
        transform = self.transform_menu.get()
        freeze_ground = self.freeze_ground_var.get()

        threading.Thread(
            target=self._stacking_thread, 
            args=(mode, feather, mask, contrast, sigma, transform, freeze_ground, gamma), 
            daemon=True
        ).start()

    def _stacking_thread(self, mode, feather, mask, contrast, sigma, transform, freeze_ground, gamma):
        def update_progress(current, total, text):
            self.after(0, lambda: self.progress_bar.set(current / total))
            self.after(0, lambda: self.status_label.configure(text=text))

        try:
            stacked, success_count, failed_reports = stack_images(
                self.image_paths, mask=mask, 
                stack_mode=mode, feather_radius=feather,
                contrast_threshold=contrast, edge_threshold=10.0, sigma=sigma,
                transform_type=transform, freeze_ground=freeze_ground, gamma=gamma,
                progress_callback=update_progress
            )
            
            if stacked is not None:
                self.output_img = stacked
                self.after(0, lambda: self._on_stacking_complete(success_count, failed_reports))
            else:
                self.after(0, lambda: self.status_label.configure(text="Stacking failed."))
                self.after(0, self._reset_ui_buttons)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {str(e)}"))
            self.after(0, lambda: self.status_label.configure(text="Stacking error occurred."))
            self.after(0, self._reset_ui_buttons)

    def _on_stacking_complete(self, success_count, failed_reports):
        self.status_label.configure(text=f"Stacking completed. {success_count} images stacked successfully.")
        self.progress_bar.set(1.0)
        
        # Hide the red mask overlay to display the clean stacked output
        self.canvas.show_mask = False
        
        # Enable controls
        self.stack_btn.configure(state="normal")
        self.load_btn.configure(state="normal")
        self.save_btn.configure(state="normal")
        self.constellation_cb.configure(state="normal")
        self.update_display_image()

        # Show detailed popup report window
        report_win = ctk.CTkToplevel(self)
        report_win.title("Stacking Run Report")
        report_win.geometry("550x450")
        report_win.attributes("-topmost", True)
        report_win.focus_force()
        
        # Center the window
        report_win.update_idletasks()
        w = report_win.winfo_width()
        h = report_win.winfo_height()
        x = (report_win.winfo_screenwidth() // 2) - (w // 2)
        y = (report_win.winfo_screenheight() // 2) - (h // 2)
        report_win.geometry(f"+{x}+{y}")

        title_lbl = ctk.CTkLabel(report_win, text="Stacking Process Finished", font=ctk.CTkFont(size=16, weight="bold"))
        title_lbl.pack(pady=10)

        stats_lbl = ctk.CTkLabel(
            report_win, 
            text=f"Total Loaded: {len(self.image_paths)}  |  Stacked Successfully: {success_count}  |  Discarded (Failed): {len(failed_reports)}"
        )
        stats_lbl.pack(pady=5)

        if failed_reports:
            err_title = ctk.CTkLabel(report_win, text="Discarded Images & Errors Details:", font=ctk.CTkFont(weight="bold"))
            err_title.pack(anchor="w", padx=20, pady=(10, 2))

            # Scrollable textbox to show details
            textbox = ctk.CTkTextBox(report_win, width=500, height=250)
            textbox.pack(padx=20, pady=5)
            
            for idx, item in enumerate(failed_reports):
                textbox.insert("end", f"[{idx+1}] File: {item['file']}\n")
                textbox.insert("end", f"    Error: {item['error']}\n\n")
            textbox.configure(state="disabled") # read-only
        else:
            success_lbl = ctk.CTkLabel(report_win, text="🎉 All images were successfully aligned and stacked!", fg_color="green", text_color="white", corner_radius=6, padx=10, pady=10)
            success_lbl.pack(pady=40)

        close_btn = ctk.CTkButton(report_win, text="Close Report", command=report_win.destroy)
        close_btn.pack(pady=15)

    def _reset_ui_buttons(self):
        self.stack_btn.configure(state="normal")
        self.load_btn.configure(state="normal")

    def toggle_constellations(self):
        self.update_display_image()

    def update_display_image(self):
        base_img = self.output_img if self.output_img is not None else self.reference_img
        if base_img is None:
            return

        if self.show_constellations_var.get():
            mask = self.canvas.get_mask()
            annotated, found = draw_constellations(base_img, mask)
            self.canvas.set_image(annotated, reset_mask=False)
            if found:
                self.status_label.configure(text="Constellation outlines drawn successfully!")
            else:
                self.status_label.configure(text="No known constellations detected in the sky region.")
        else:
            self.canvas.set_image(base_img, reset_mask=False)
            self.status_label.configure(text="Clean image displayed.")

    def save_result(self):
        if self.output_img is None:
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Stacked Image",
            defaultextension=".tiff",
            filetypes=[("TIFF Image", "*.tiff *.tif"), ("PNG Image", "*.png"), ("JPEG Image", "*.jpg *.jpeg")]
        )
        if file_path:
            cv2.imwrite(file_path, self.output_img)
            self.status_label.configure(text=f"Saved stacked image to: {os.path.basename(file_path)}")
            messagebox.showinfo("Saved", "Image saved successfully.")

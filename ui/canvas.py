import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2

class MaskingCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.original_cv_img = None
        self.display_image = None
        self.tk_image = None
        
        # Mask parameters
        self.mask_img = None       # PIL Image for mask drawing
        self.mask_draw = None      # ImageDraw object
        
        # Drawing parameters
        self.brush_size = 30
        self.brush_color = "red"   # Visual representation in GUI
        self.is_painting = False
        self.draw_mode = "add"     # "add" to paint sky (255), "erase" to remove sky (0)
        self.show_mask = True      # Toggle to show/hide the red sky mask overlay

        # Scale factors to map canvas coordinates to original image coordinates
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # Event bindings
        self.bind("<Configure>", self.on_resize)
        
        # Left click/drag to PAINT sky (255)
        self.bind("<Button-1>", lambda e: self.start_paint(e, "add"))
        self.bind("<B1-Motion>", lambda e: self.paint(e, "add"))
        self.bind("<ButtonRelease-1>", self.stop_paint)
        
        # Right click/drag to ERASE sky (0)
        self.bind("<Button-3>", lambda e: self.start_paint(e, "erase"))
        self.bind("<B3-Motion>", lambda e: self.paint(e, "erase"))
        self.bind("<ButtonRelease-3>", self.stop_paint)
        
        # Mouse Wheel to resize brush
        self.bind("<MouseWheel>", self.on_mouse_wheel)
        self.bind("<Motion>", self.draw_brush_cursor)

        # Callback for when brush size changes via scroll wheel
        self.on_brush_size_changed = None
        self.paint_mode = "add"

        # Cursor circle reference
        self.cursor_circle = None

    def set_image(self, cv_img, reset_mask=True):
        """Loads a new reference image and optionally resets the mask."""
        self.original_cv_img = cv_img
        h, w, _ = cv_img.shape
        
        if reset_mask or self.mask_img is None:
            # Create blank mask (all black = ground by default)
            self.mask_img = Image.new("L", (w, h), 0)
            self.mask_draw = ImageDraw.Draw(self.mask_img)
        
        self.redraw()

    def set_brush_size(self, size):
        self.brush_size = int(size)

    def set_draw_mode(self, mode):
        """mode: 'add' (paint sky, 255) or 'erase' (paint ground, 0)"""
        self.draw_mode = mode
        if mode == "add":
            self.brush_color = "red"
        else:
            self.brush_color = "blue"

    def clear_mask(self):
        if self.original_cv_img is not None:
            h, w, _ = self.original_cv_img.shape
            self.mask_img = Image.new("L", (w, h), 0)
            self.mask_draw = ImageDraw.Draw(self.mask_img)
            self.redraw()

    def get_mask(self):
        """Returns the mask as a numpy array matching the original image size."""
        if self.mask_img is None:
            return None
        return np.array(self.mask_img)

    def on_resize(self, event):
        self.redraw()

    def redraw(self):
        if self.original_cv_img is None:
            self.delete("all")
            self.create_text(
                self.winfo_width() / 2, self.winfo_height() / 2,
                text="No image loaded. Load images to begin.",
                fill="gray", font=("Arial", 14)
            )
            return

        self.delete("all")

        # Get canvas dimensions
        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return

        # Original image dimensions
        img_h, img_w, _ = self.original_cv_img.shape

        # Calculate scale factor to fit canvas
        scale_w = canvas_w / img_w
        scale_h = canvas_h / img_h
        self.scale_factor = min(scale_w, scale_h)

        # New display dimensions
        disp_w = int(img_w * self.scale_factor)
        disp_h = int(img_h * self.scale_factor)

        # Offsets to center the image on the canvas
        self.offset_x = (canvas_w - disp_w) // 2
        self.offset_y = (canvas_h - disp_h) // 2

        # Convert OpenCV BGR to PIL RGB
        rgb_img = cv2.cvtColor(self.original_cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # Resize background image
        resized_img = pil_img.resize((disp_w, disp_h), Image.Resampling.LANCZOS)

        # Create overlay for the mask if enabled
        if self.show_mask:
            # Resize mask image
            resized_mask = self.mask_img.resize((disp_w, disp_h), Image.Resampling.NEAREST)
            
            # Colorize the mask (Red overlay for sky, semi-transparent)
            overlay = Image.new("RGBA", (disp_w, disp_h), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            mask_np = np.array(resized_mask)
            # Red where sky is drawn (mask == 255)
            overlay_data = np.zeros((disp_h, disp_w, 4), dtype=np.uint8)
            overlay_data[mask_np == 255] = [255, 0, 0, 100]  # Red overlay with alpha=100
            
            overlay = Image.fromarray(overlay_data, "RGBA")
            
            # Composite background and overlay
            composite = Image.alpha_composite(resized_img.convert("RGBA"), overlay)
        else:
            composite = resized_img.convert("RGBA")
            
        self.tk_image = ImageTk.PhotoImage(composite)
        self.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_image)

    def start_paint(self, event, mode):
        self.show_mask = True  # Restore mask visibility when starting to paint
        self.is_painting = True
        self.paint_mode = mode
        self.paint(event, mode)

    def stop_paint(self, event):
        self.is_painting = False

    def paint(self, event, mode=None):
        if self.original_cv_img is None or not self.is_painting:
            return

        if mode is None:
            mode = self.paint_mode

        # Check if click is inside the image bounds
        x = event.x - self.offset_x
        y = event.y - self.offset_y

        img_h, img_w, _ = self.original_cv_img.shape
        disp_w = int(img_w * self.scale_factor)
        disp_h = int(img_h * self.scale_factor)

        if 0 <= x < disp_w and 0 <= y < disp_h:
            # Map back to original resolution
            orig_x = int(x / self.scale_factor)
            orig_y = int(y / self.scale_factor)
            orig_brush = int(self.brush_size / self.scale_factor)

            val = 255 if mode == "add" else 0
            
            # Draw circle on original resolution mask
            self.mask_draw.ellipse(
                [orig_x - orig_brush, orig_y - orig_brush, orig_x + orig_brush, orig_y + orig_brush],
                fill=val
            )
            
            self.redraw()
            self.draw_brush_cursor(event)

    def on_mouse_wheel(self, event):
        if self.original_cv_img is None:
            return
        # event.delta is positive when scrolling up, negative when scrolling down
        if event.delta > 0:
            self.brush_size = min(100, self.brush_size + 2)
        else:
            self.brush_size = max(5, self.brush_size - 2)

        if self.on_brush_size_changed:
            self.on_brush_size_changed(self.brush_size)
            
        self.draw_brush_cursor(event)

    def draw_brush_cursor(self, event):
        """Draws a visual guide representing the brush outline."""
        if self.original_cv_img is None:
            return

        if self.cursor_circle:
            self.delete(self.cursor_circle)

        # Check image boundaries
        x = event.x
        y = event.y

        # Determine outline color based on whether we are drawing or hovering
        if self.is_painting:
            color = "red" if self.paint_mode == "add" else "blue"
        else:
            color = "white"

        # Draw circle on canvas
        r = self.brush_size
        self.cursor_circle = self.create_oval(
            x - r, y - r, x + r, y + r,
            outline=color, width=2
        )

    def save_mask_to_file(self, file_path):
        if self.mask_img is not None:
            self.mask_img.save(file_path)
            return True
        return False

    def load_mask_from_file(self, file_path):
        if self.original_cv_img is None:
            return False
        try:
            loaded_img = Image.open(file_path).convert("L")
            h, w, _ = self.original_cv_img.shape
            if loaded_img.size != (w, h):
                loaded_img = loaded_img.resize((w, h), Image.Resampling.NEAREST)
            self.mask_img = loaded_img
            self.mask_draw = ImageDraw.Draw(self.mask_img)
            self.redraw()
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load mask: {e}")
            return False

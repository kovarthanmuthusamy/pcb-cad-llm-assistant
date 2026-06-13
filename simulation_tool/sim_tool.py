import os,time
from datetime import datetime
from pathlib import Path
import pyautogui,subprocess
import pygetwindow as gw
import win32gui,win32con  # type: ignore


def set_window_on_top(title, pin=True):
    """Pin/unpin a window."""
    windows = [win for win in gw.getWindowsWithTitle(title) if win]
    if windows:
        hwnd = windows[0]._hWnd
        if pin:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        else:
            win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        return True
    return False


def move_window_offscreen(title):
    """Moves a window off-screen instead of minimizing it."""
    windows = [win for win in gw.getWindowsWithTitle(title) if win]
    if windows:
        hwnd = windows[0]._hWnd
        win32gui.SetWindowPos(hwnd, 0, -5000, -5000, 500, 500, win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE)


def simulation(type="", topology="P2P", L_value=None, R_value=None):
    """Runs eCADSTAR simulation while keeping VS Code pinned on top."""
    simulation_type = type  # Fix: Assign the parameter instead of `type`

    # Choose files for corresponding topology
    topo_paths = {
        "P2P": (os.getenv("ECADSTAR_P2P_SIDF"), os.getenv("ECADSTAR_P2P_TEMPLATE")),
        "Star": (os.getenv("ECADSTAR_STAR_SIDF"), os.getenv("ECADSTAR_STAR_TEMPLATE")),
    }

    if topology not in topo_paths:
        print("Error: Topology not found")
        return

    filepath_simulation_sidf, filepath_simulation_txt = topo_paths[topology]
    if not filepath_simulation_sidf or not filepath_simulation_txt:
        raise RuntimeError(
            f"eCADSTAR paths for {topology} are not configured; see .env.example"
        )

    t0 = time.time()  # Start time

    try:
       # vscode_was_open = set_window_on_top("Visual Studio Code", pin=True)  # Pin VS Code

        # Update SIDF file with R & L values
        if L_value is not None or R_value is not None:
            with open(filepath_simulation_txt, "r") as file:
                txt_string = file.read()

            if R_value is not None:
                txt_string = txt_string.replace("RESISTOR_VALUE", str(int(R_value * 100)))

            if L_value is not None:
                txt_string = txt_string.replace("LENGTH_VALUE", str(int(L_value * 100000)))

            with open(filepath_simulation_sidf, "w") as file:
                file.write(txt_string)

        os.startfile(filepath_simulation_sidf)  # Open manipulated simulation file
        time.sleep(10)  # Wait for opening

        # Move eCADSTAR off-screen to hide UI interactions
       # move_window_offscreen("eCADSTAR")

        # Handle pop-ups (locked simulation file or accept changes)
        pyautogui.press("enter")
        time.sleep(1)

        # Select simulation type using hotkeys
        sim_hotkeys = {"SI": ["alt", "t"], "TDR": ["alt", "t", "down"], "FD": ["alt", "t", "down", "down", "down"]}
        if simulation_type in sim_hotkeys:
            for key in sim_hotkeys[simulation_type]:
                pyautogui.press(key)
            pyautogui.press("enter")
        else:
            print("Error: Simulation type not found")
            return

        time.sleep(10)  # Wait for simulation to finish
        subprocess.call(["taskkill", "/F", "/IM", "eCS_eleceditor.exe"])
        # Call image exporter function (ensure it's defined)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"image_{timestamp}.png"
        output_dir = Path(
            os.getenv(
                "ECADSTAR_OUTPUT_DIR",
                str(Path(__file__).resolve().parent / "simulation_images"),
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        img_path = str(output_dir / file_name)
        print(img_path)
        # Export as an image
        pyautogui.hotkey('alt', 'f')
        pyautogui.hotkey("down")
        pyautogui.hotkey("enter")
        time.sleep(0.5)
        pyautogui.hotkey("down")
        time.sleep(0.5)
        pyautogui.hotkey("tab")
        pyautogui.hotkey("backspace")
        time.sleep(1)
        pyautogui.typewrite(img_path)
        time.sleep(1)
        pyautogui.hotkey("enter")
        time.sleep(1)
        pyautogui.hotkey("enter")
        time.sleep(1)
        pyautogui.hotkey("enter")
        time.sleep(0.5)
        pyautogui.hotkey("enter") 
        time.sleep(1)
        pyautogui.hotkey('winleft', 'down')
        # Kill eCADSTAR and viewer windows
       # subprocess.call(["taskkill", "/F", "/IM", "viewer.exe"])

        t1 = time.time()  # End time

        print(f"Simulation completed in {round(t1 - t0, 1)} seconds")
        return img_path

    finally:
       # if vscode_was_open:
        #    set_window_on_top("Visual Studio Code", pin=False)  # Restore VS Code
        print("hello")

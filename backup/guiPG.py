import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.configure_app(manual_callback_management=True)
dpg.create_viewport()
dpg.setup_dearpygui()

def callback(sender, app_data, user_data):
    print("Called on the main thread!")

with dpg.window(label="Tutorial"):
    dpg.add_button(label="Press me", callback=callback)


# main loop
dpg.show_viewport()
while dpg.is_dearpygui_running():
    jobs = dpg.get_callback_queue() # retrieves and clears queue
    dpg.run_callbacks(jobs)
    dpg.render_dearpygui_frame()

dpg.destroy_context()
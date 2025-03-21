import gradio as gr
from function.history import  load_i2v_history
from function.video_generation import generate_video_from_image
from function.open_output_folder import i2v_open_output_folder
def setup_i2v_button_events(
    run_i2v_button, run_i2v_button_Disable, i2v_prompt, i2v_negative_prompt, i2v_input_image, i2v_input_video,
    i2v_denoising_strength, i2v_seed, i2v_rand_device, i2v_resolution, i2v_num_frames, i2v_cfg_scale,
    i2v_num_inference_steps, i2v_sigma_shift, i2v_tiled, i2v_tile_size, i2v_tile_stride,i2v_output_fps, i2v_output_quality,
    i2v_result_gallery, i2v_history, i2v_open_folder_button
):
    i2v_generation_state = gr.Checkbox(value=False, visible=False)

    # 定义切换 generation_state 值的函数
    def i2v_toggle_generation_state(current_state):
        return not current_state

    # 定义函数，用于切换生成按钮
    def toggle_run_i2v_button():
        # 视频生成完成后，隐藏禁用按钮，显示生成按钮
        return gr.update(visible=False), gr.update(visible=True)

    def toggle_run_i2v_button_Disable():
        # 点击生成按钮后，隐藏生成按钮，显示禁用按钮
        return gr.update(visible=False), gr.update(visible=True)

    # 修改生成按钮的点击事件
    run_i2v_button.click(
        fn=toggle_run_i2v_button_Disable,
        outputs=[run_i2v_button, run_i2v_button_Disable]
    ).then(
        fn=generate_video_from_image,
        inputs=[
            i2v_prompt,
            i2v_negative_prompt,
            i2v_input_image,
            i2v_input_video,
            i2v_denoising_strength,
            i2v_seed,
            i2v_rand_device,
            i2v_resolution,
            i2v_num_frames,
            i2v_cfg_scale,
            i2v_num_inference_steps,
            i2v_sigma_shift,
            i2v_tiled,
            i2v_tile_size,
            i2v_tile_stride,
            i2v_output_fps,
            i2v_output_quality
        ],
        outputs=[i2v_result_gallery],
        show_progress="full",
    ).then(
        fn=toggle_run_i2v_button,
        outputs=[run_i2v_button_Disable, run_i2v_button]
    ).then(
        fn=i2v_toggle_generation_state,
        inputs=[i2v_generation_state],
        outputs=[i2v_generation_state]
    )

    def i2v_update_examples():
        i2v_history_list = load_i2v_history()
        return gr.Dataset(samples=i2v_history_list)

    i2v_generation_state.change(i2v_update_examples, None, i2v_history.dataset)


    # 绑定按钮点击事件
    i2v_open_folder_button.click(
        fn=i2v_open_output_folder
    )

    return i2v_generation_state

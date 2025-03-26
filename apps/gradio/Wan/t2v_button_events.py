import gradio as gr
from function.history import load_t2v_history
from function.video_generation import generate_video_from_text
from function.open_output_folder import t2v_open_output_folder 
from function.open_prompt_reference import open_prompt_reference
from function.prompt_refiners import optimize_chinese_prompt
def setup_t2v_button_events(
    run_t2v_button, run_t2v_button_Disable, t2v_prompt, t2v_negative_prompt, t2v_input_image, t2v_input_video,
    t2v_denoising_strength, t2v_seed, t2v_rand_device, t2v_resolution, t2v_num_frames, t2v_cfg_scale,
    t2v_num_inference_steps, t2v_sigma_shift, t2v_tiled, t2v_tile_size, t2v_tile_stride, output_fps, output_quality,
    result_gallery, t2v_history, t2v_open_folder_button,t2v_prompt_reference_button,
        t2v_prompt_refiner_button,t2v_ModelChoices,t2v_num_persistent_param_in_dit,t2v_loadChoices
):
    generation_state = gr.Checkbox(value=False, visible=False)

    # 定义切换 generation_state 值的函数
    def toggle_generation_state(current_state):
        return not current_state

    # 定义函数，用于切换生成按钮
    def toggle_run_t2v_button():
        # 视频生成完成后，隐藏禁用按钮，显示生成按钮
        return gr.update(visible=False), gr.update(visible=True)

    def toggle_run_t2v_button_Disable():
        # 点击生成按钮后，隐藏生成按钮，显示禁用按钮
        return gr.update(visible=False), gr.update(visible=True)

    # 修改生成按钮的点击事件
    run_t2v_button.click(
        fn=toggle_run_t2v_button_Disable,
        outputs=[run_t2v_button, run_t2v_button_Disable]
    ).then(
        fn=generate_video_from_text,
        inputs=[
            t2v_prompt,
            t2v_negative_prompt,
            t2v_input_image,
            t2v_input_video,
            t2v_denoising_strength,
            t2v_seed,
            t2v_rand_device,
            t2v_resolution,
            t2v_num_frames,
            t2v_cfg_scale,
            t2v_num_inference_steps,
            t2v_sigma_shift,
            t2v_tiled,
            t2v_tile_size,
            t2v_tile_stride,
            output_fps,
            output_quality,
            t2v_ModelChoices,
            t2v_num_persistent_param_in_dit,
            t2v_loadChoices
        ],
        outputs=[result_gallery],
        show_progress="full",
    ).then(
        fn=toggle_run_t2v_button,
        outputs=[run_t2v_button_Disable, run_t2v_button]
    ).then(
        fn=toggle_generation_state,
        inputs=[generation_state],
        outputs=[generation_state]
    )

    def update_examples():
        t2v_history_list = load_t2v_history()
        return gr.Dataset(samples=t2v_history_list)

    generation_state.change(update_examples, None, t2v_history.dataset)


    # 绑定按钮点击事件
    t2v_open_folder_button.click(
        fn=t2v_open_output_folder
    )
    t2v_prompt_reference_button.click(
        fn=open_prompt_reference
    )

    t2v_prompt_refiner_button.click(
        fn=optimize_chinese_prompt, inputs=[t2v_prompt], outputs=[t2v_prompt]
    )

    return generation_state

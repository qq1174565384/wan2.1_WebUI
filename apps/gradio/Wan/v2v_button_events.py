import gradio as gr
from function.history import  load_v2v_history
from function.video_generation import generate_video_from_image
from function.open_output_folder import v2v_open_output_folder
from function.prompt_refiners import optimize_chinese_prompt 
from function.prompt_inference import prompt_inference
from function.prompt_translate import prompt_translate
def setup_v2v_button_events(
    run_v2v_button, run_v2v_button_Disable, v2v_prompt, v2v_negative_prompt, v2v_input_image, v2v_input_video,
    v2v_denoising_strength, v2v_seed, v2v_rand_device, v2v_resolution, v2v_num_frames, v2v_cfg_scale,
    v2v_num_inference_steps, v2v_sigma_shift, v2v_tiled, v2v_tile_size, v2v_tile_stride,v2v_output_fps, v2v_output_quality,
    v2v_result_gallery, v2v_history, v2v_open_folder_button,v2v_num_persistent_param_in_dit,v2v_ModelChoices,
        v2v_loadChoices,
        v2v_prompt_refiner_button,
        v2v_prompt_inference_button
):
    v2v_generation_state = gr.Checkbox(value=False, visible=False)

    # 定义切换 generation_state 值的函数
    def v2v_toggle_generation_state(current_state):
        return not current_state

    # 定义函数，用于切换生成按钮
    def toggle_run_v2v_button():
        # 视频生成完成后，隐藏禁用按钮，显示生成按钮
        return gr.update(visible=False), gr.update(visible=True)

    def toggle_run_v2v_button_Disable():
        # 点击生成按钮后，隐藏生成按钮，显示禁用按钮
        return gr.update(visible=False), gr.update(visible=True)

    # 修改生成按钮的点击事件
    run_v2v_button.click(
        fn=toggle_run_v2v_button_Disable,
        outputs=[run_v2v_button, run_v2v_button_Disable]
    ).then(
        fn=generate_video_from_image,
        inputs=[
            v2v_prompt,
            v2v_negative_prompt,
            v2v_input_image,
            v2v_input_video,
            v2v_denoising_strength,
            v2v_seed,
            v2v_rand_device,
            v2v_resolution,
            v2v_num_frames,
            v2v_cfg_scale,
            v2v_num_inference_steps,
            v2v_sigma_shift,
            v2v_tiled,
            v2v_tile_size,
            v2v_tile_stride,
            v2v_output_fps,
            v2v_output_quality,
            v2v_num_persistent_param_in_dit,
            v2v_ModelChoices,
            v2v_loadChoices
        ],
        outputs=[v2v_result_gallery],
        show_progress="full",
    ).then(
        fn=toggle_run_v2v_button,
        outputs=[run_v2v_button_Disable, run_v2v_button]
    ).then(
        fn=v2v_toggle_generation_state,
        inputs=[v2v_generation_state],
        outputs=[v2v_generation_state]
    )

    def v2v_update_examples():
        v2v_history_list = load_v2v_history()
        return gr.Dataset(samples=v2v_history_list)

    v2v_generation_state.change(v2v_update_examples, None, v2v_history.dataset)


    # 绑定按钮点击事件
    v2v_open_folder_button.click(
        fn=v2v_open_output_folder
    )
    # 绑定按钮点击事件
    v2v_prompt_inference_button.click(
        fn=prompt_inference, inputs=[v2v_input_image], outputs=[v2v_prompt]
    ).then(
        fn=prompt_translate, inputs=[v2v_prompt], outputs=[v2v_prompt]
    )
    # 绑定按钮点击事件
    v2v_prompt_refiner_button.click(
        fn=optimize_chinese_prompt, inputs=[v2v_prompt], outputs=[v2v_prompt]
    )
 



    return v2v_generation_state

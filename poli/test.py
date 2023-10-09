from data_process import model_download,step2_selection,generate_ft_data

# large_model_name = "meta-llama/Llama-2-7b-chat-hf"
sizes = ["small","base","large"]
for size in sizes:
    print(f"{size} =================================================================")
    small_model_name = f"google/flan-t5-{size}"
    output_dir = f"T5filter_{size}"
    # model_download(small_model_name)
    step2_selection("self_consistency1",small_model_name,output_dir)
    generate_ft_data("qasc",output_dir)


exit()














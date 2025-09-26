from runLDAfromDF import runLDAfromDFusingConfigMenu
import time

if __name__=="__main__":

    # LDA配置文件
    lda_cfg={
        "runtime_code":"cyj", # 运行时代码，必选项，用于区分不同的运行时，用于输出文件名的开头，应该是字符串格式
        "input_file":"example_output_wt.pkl", # 输入文件，必选项，应该是一个pandas.DataFrame或内容是一个DataFrame的pickle文件的路径名
        "tokenized_column_name":"NewsContentTokenized", # 分词结果的列的列名，必选项，应该是字符串格式，内容应该是以空格分隔的token
        "text_feature_extractor":"bow", # 构建文本特征所用的模型，可选项，默认为"bow"词袋模型，也可以选择"tfidf"模型
        "start_num_topic":20, # 开始主题数，可选项，默认为4，程序会遍历range(start_num_topic,end_num_topic,num_topic_step)完成不同主题数的LDA运算
        "end_num_topic":120, # 结束主题数，可选项，默认为96，程序会遍历range(start_num_topic,end_num_topic,num_topic_step)完成不同主题数的LDA运算
        "num_topic_step":4, # 主题数步长，可选项，默认为4，程序会遍历range(start_num_topic,end_num_topic,num_topic_step)完成不同主题数的LDA运算
        "output_filename":"example_output_lda.pkl", # 输出文件名，可选项，默认为None即不需要输出，为防止文本中逗号与逗号分隔符混淆不支持csv，支持xlsx，数据量较大时会以1000000行为分割输出多个excel文件，支持pickle(.pkl)格式，输出文件会出现在finalresults文件夹中
        # 以下两项respwanpoint文件夹配置，respawnpoint文件夹用于存储临时文件，以避免在读取大量数据时占用过多内存造成程序崩溃，若不清空respawnpoint文件夹可能导致之前的临时文件被重复读入
        "clear_respawnpoint_before_run":False, # 开始运行前是否清空respawnpoint文件夹，可选项，默认为False，若无需使用上一次运行的结果或希望在运行前保持respwanpoint文件夹的清洁，可以设为True
        "clear_respawnpoint_upon_conplete":False # 完成后是否清空respawnpoint文件夹，可选项，默认为True，若在下一次运行时还需要使用本次运行的部分结果，请设置为False
    }

    print("程序运行开始\n")
    t0=time.time()
    rtn_lda=runLDAfromDFusingConfigMenu(config_menu=lda_cfg)
    print(rtn_lda.head(10))
    print(f"\n程序运行完成，用时{(time.time()-t0):.4f}秒")
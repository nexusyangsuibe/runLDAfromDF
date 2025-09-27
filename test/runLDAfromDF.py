# LDA模块

from collections.abc import Iterable
from hashlib import sha256
import multiprocessing as mp
from pathlib import Path
from io import StringIO
import pickle
import time
import os
import re

import pandas as pd
import numpy as np
import xlsxwriter

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# common tool functions are as follows
def ensureCorrectPklDump(obj,filepath):
    # ensure that objects written into pickle files can be read normally to avoid corrupted writing
    fail=0
    path=Path(filepath)
    pathname=path.parent
    filename=path.name
    pickle.dump(obj,open(pathname/f"tmp_{filename}", "wb"))
    while True:
        if fail>2:
            raise RuntimeError(f"写入pickle文件已经失败了{fail}次，请检查写入对象的完整性")
        try:
            pickle.load(open(pathname/f"tmp_{filename}","rb"))
            break
        except:
            fail+=1
            pickle.dump(obj,open(pathname/f"tmp_{filename}","wb"))
    if os.path.exists(path):
        os.remove(path)
    os.rename(pathname/f"tmp_{filename}",path)
    return None

def findBestBulkNum(df,thereshold_GB,best_bulk_num=1):
    # find the best bulk number that meets the demand that all bulks smaller than thereshold_GB
    for idx in range(best_bulk_num):
        memory_usage_GB=df.iloc[int(len(df)*(idx/best_bulk_num)):int(len(df)*((idx+1)/best_bulk_num))].memory_usage(deep=True).sum()/(1024**3)
        if memory_usage_GB>thereshold_GB:
            new_bulk_num=max(int(df.memory_usage(deep=True).sum()/(1024**3))//thereshold_GB+1,best_bulk_num+1)
            return findBestBulkNum(df,thereshold_GB,best_bulk_num=new_bulk_num)
    else:
        return best_bulk_num

def outputAccording2BestBulkNum(param):
    # write into excel according to the best bulk number
    df_bulk,fileName,file_rows,thereshold_GB=param
    df_bulk=df_bulk.map(lambda x: str(x) if isinstance(x,Iterable) and not isinstance(x,str) else x)
    bulk_num=findBestBulkNum(df_bulk,thereshold_GB)
    if bulk_num==1:
        workbook=xlsxwriter.Workbook(fileName,{'constant_memory':True,"strings_to_urls":False,"nan_inf_to_errors":True})
        worksheet=workbook.add_worksheet()
        worksheet.write_row(0,0,df_bulk.columns)
        for row_idx,row in enumerate(df_bulk.itertuples(index=False),start=0):
            worksheet.write_row(row_idx+1,0,row)
        workbook.close()
    else:
        print(f"文件{fileName}所需的存储空间超过阙值{thereshold_GB}GB，再分为{bulk_num}个文件输出")
        for iidx in range(bulk_num):
            fileName_=f"{''.join(fileName.split('.')[:-1])}_{iidx}.xlsx"
            print(f"正在写入{fileName_}")
            workbook=xlsxwriter.Workbook(fileName_,{'constant_memory':True,"strings_to_urls":False,"nan_inf_to_errors":True})
            worksheet=workbook.add_worksheet()
            worksheet.write_row(0,0,df_bulk.columns)
            for row_idx,row in enumerate(df_bulk.iloc[int(file_rows*(iidx/bulk_num)):int(file_rows*((iidx+1)/bulk_num))].itertuples(index=False),start=0):
                worksheet.write_row(row_idx+1,0,row)
            workbook.close()
    return None

def outputAsXlsx(df,output_filename,output_pathname,thereshold_rows=1000000,thereshold_GB=4):
    # output the dataframe into excel with divsions within the thereshold_rows and thereshold_GB
    file_num=int(df.shape[0]//thereshold_rows)
    print(f"共{df.shape[0]}行，文件名为{output_filename}，预计分为{file_num+1}个文件输出")
    if file_num==0:
        outputAccording2BestBulkNum((df,f"{output_pathname}{'' if output_pathname.endswith('/') else '/'}{''.join(output_filename.split('.')[:-1])}.xlsx",None,thereshold_GB))
    else:
        file_rows,last_rows=divmod(df.shape[0],file_num+1)
        last_rows=file_rows+last_rows
        print(f"每个文件约有{file_rows}行")
        tasks=[]
        for idx in range(file_num):
            df_bulk=df.iloc[idx*file_rows:(idx+1)*file_rows]
            fileName=f"{output_pathname}{'' if output_pathname.endswith('/') else '/'}{''.join(output_filename.split('.')[:-1])}_{idx}.xlsx"
            tasks.append((df_bulk,fileName,file_rows,thereshold_GB))
        if last_rows:
            df_bulk=df.iloc[file_num*file_rows:]
            fileName=f"{output_pathname}{'' if output_pathname.endswith('/') else '/'}{''.join(output_filename.split('.')[:-1])}_{file_num}.xlsx"
            tasks.append((df_bulk,fileName,file_rows,thereshold_GB))
        pool=mp.Pool(processes=8)
        pool.map(outputAccording2BestBulkNum,tasks)
    return None

def saveConcatedDataAsFinalResult(runtime_code,concatedDF,output_filename,clear_respawnpoint_upon_conplete):
    # the end process of the concatDF, including writing the final result to the disk and clear the respawnpoint folder
    if not clear_respawnpoint_upon_conplete or not output_filename:
        ensureCorrectPklDump(concatedDF,f"respawnpoint/{runtime_code}_word_tokenized.pkl")
    if output_filename:
        print("开始将最终结果写入硬盘")
        if output_filename.endswith(".pkl"):
            ensureCorrectPklDump(concatedDF,f"finalresults/{output_filename}")
        elif output_filename.endswith(".xlsx"):
            outputAsXlsx(concatedDF,output_filename,"finalresults")
        elif output_filename.endswith(".csv"):
            concatedDF.to_csv(f"finalresults/{output_filename}")
        else:
            raise ValueError(f"不支持的文件格式{output_filename}，请核查")
    if clear_respawnpoint_upon_conplete:
        if not output_filename and input("由于未指定output_filename，finalresults文件夹中不会产生任何结果文件，又clear_respawnpoint_upon_conplete参数为True，将清空respawnpoint文件夹中的所有临时文件，因此您的本次运行不会产生任何可观测的结果，输入y继续，输入其他任意字符取消清空respawnpoint文件夹：").lower()!="y":
            print("用户取消清空respawnpoint文件夹")
            return False # return False to indicate that nothing emerge in either finalresults or respawnpoint
        print("开始清空respawnpoint文件夹")
        for file in os.listdir("respawnpoint/"):
            os.remove("respawnpoint/" + file)
    return None

def runLDAwithRetry(runtime_code,num_topic,text_features,df_identity_code,tokenized_column_name,text_feature_extractor,n_jobs,retry=0):
    # run lda with diminishing n_jobs until success or n_jobs==0
    start_time=time.time()
    print(f"第{retry+1}次计算{num_topic}主题数，线程数{n_jobs}，结果",end="",flush=True)
    # code not using try-except for debug
    # perplexity_training=run_lda_once(runtime_code,num_topic,text_features,df_identity_code,tokenized_column_name,text_feature_extractor,n_jobs)
    # print(f"成功，用时{round(time.time()-start_time,4)}秒，样本内困惑度{perplexity_training}")
    # return perplexity_training,n_jobs
    try:
        perplexity_training=run_lda_once(runtime_code,num_topic,text_features,df_identity_code,tokenized_column_name,text_feature_extractor,n_jobs)
        print(f"成功，用时{round(time.time()-start_time,4)}秒，样本内困惑度{perplexity_training}")
        if n_jobs>1:
            n_jobs=n_jobs-1 if np.random.random()<0.7 else n_jobs
        return perplexity_training,n_jobs
    except OSError as e:
        print(f"失败，可用资源不足，",end="")
    except MemoryError as e:
        print(f"失败，内存不足，",end="")
    except Exception as e:
        print(f"失败，{e}，",end="")
    time_consumed=round(time.time()-start_time,4)
    print(f"用时{time_consumed}秒，失败是在寻找最优线程数时的正常结果请不必担心")
    retry+=1 # you should not write 'finally' before this line else the finally block will run no matter the try block returns or not
    n_jobs=int(n_jobs*0.8) if n_jobs>os.cpu_count()*0.3 else n_jobs-1
    if n_jobs<1:
        raise RuntimeError("已经将进程数降低为单进程但依然没有足够的系统资源完成任务，请尝试更换更好的设备后重试")
    return runLDAwithRetry(runtime_code,num_topic,text_features,df_identity_code,tokenized_column_name,text_feature_extractor,n_jobs,retry)

def run_lda_once(runtime_code,num_topic,text_features,df_identity_code,tokenized_column_name,text_feature_extractor,n_jobs):
    # the workhorse for running lda
    lda=LatentDirichletAllocation(n_components=num_topic,learning_method="batch",max_iter=24,max_doc_update_iter=240,n_jobs=n_jobs) # I want to add an option to use 'online' learning method, but it me encounter some weird mistakes like fail to pickle so I forgive it
    lda_dtf=lda.fit_transform(text_features)
    perplexity=int(lda.bound_)
    ensureCorrectPklDump(tuple([lda,lda_dtf]),f"respawnpoint/{runtime_code}_lda_{num_topic}_{perplexity}_{df_identity_code}_{tokenized_column_name}_{text_feature_extractor}.pkl")
    return perplexity

def runLDAfromDF(runtime_code,input_file,tokenized_column_name,text_feature_extractor,start_num_topic,end_num_topic,num_topic_step,output_filename,clear_respawnpoint_before_run,clear_respawnpoint_upon_conplete):
    print("LDA模块开始运行")
    # create the folders if not exists
    if "respawnpoint" not in os.listdir():
        print(f"在工作目录{os.getcwd()}下未找到用于存储临时文件的respawnpoint文件夹，将自动创建")
        os.mkdir("respawnpoint")
    if "finalresults" not in os.listdir():
        print(f"在工作目录{os.getcwd()}下未找到用于存储最终结果的finalresults文件夹，将自动创建")
        os.mkdir("finalresults")
    # read and check the input file
    if type(input_file)==str and input_file.endswith(".pkl"):
        input_file=pickle.load(open(input_file,"rb"))
    if type(input_file)!=pd.DataFrame:
        raise ValueError("输入文件必须是DataFrame类型或包含DataFrame的pickle文件路径名")
    if input_file.shape[0]==0:
        raise ValueError("输入的DataFrame为空")
    tmp_buf=StringIO()
    input_file.info(buf=tmp_buf)
    df_identity_code=sha256(tmp_buf.getvalue().encode(encoding="utf-8")).hexdigest()[:24] # used to check the whether the file in the respawnpoint is the results of handling the same DataFrame
    text_feature_extractor=text_feature_extractor.lower()
    if text_feature_extractor=="bow":
        vectorizer=CountVectorizer(ngram_range=(1,1))
    elif text_feature_extractor in ["tfidf","tfidf"]:
        text_feature_extractor="tfidf" # omit hyfen to prettify the tmp filename
        vectorizer=TfidfVectorizer(ngram_range=(1,1))
    else:
        raise ValueError(f"'text_feature_extractor'参数只允许传入'bow'或'tfidf'，用户传入的{text_feature_extractor=}不合理")
    # check the respawnpoint to find the finished and unfinished interval
    finished_num_topics=[]
    result_filenames=[]
    for raw_filename in os.listdir("respawnpoint"):
        if ck_point:=re.match(f"{runtime_code}_lda_(\\d+)_(\\d+)_{df_identity_code}_{tokenized_column_name}_{text_feature_extractor}.pkl",raw_filename):
            result_filenames.append("respawnpoint/"+raw_filename)
            finished_num_topics.append([int(ck_point.group(1)),int(ck_point.group(2))]) # the first number is n_components, the second number is int(perplexity)
    # clear the respawnpoint folder before running if needed
    if clear_respawnpoint_before_run:
        if not finished_num_topics or input(f"您输入的参数{clear_respawnpoint_before_run=}要求在运行前删去respawnpoint文件夹中的所有内容，但程序在该文件夹中检测到已经完成运行的主题数{finished_num_topics=}，若按计划删去respawnpoint文件夹中的所有内容则已完成分词部分的区间记录将被删除且无法恢复，即程序将完全从头开始重新分词而不是继续已经完成的部分，输入y确认，输入其他任意字符取消清空respawnpoint文件夹：").lower()=="y":
            for file in os.listdir("respawnpoint/"):
                os.remove("respawnpoint/"+file)
            finished_num_topics=[]
    # continue to process the DataFrame
    index_name=input_file.index.name # save the index name to restore it after reset the index
    if not index_name:
        while True: # if the existing index does not have a name, we need to give it one name else we cannnot recover it
            new_index_name=f"neue_idx_name_{np.random.randint(10000,100000)}"
            if new_index_name not in input_file.columns:
                break
        input_file.index.name=new_index_name
        index_name=new_index_name
    input_file=input_file.reset_index() # reset the index to default increasing primary key to ensure that the index is unique
    ensureCorrectPklDump(input_file,f"respawnpoint/{runtime_code}_input_dataframe_backup.pkl")
    if type(tokenized_column_name)!=str or tokenized_column_name not in input_file.columns:
        raise ValueError(f"输入的tokenized_column必须是字符串类型且存在于输入文件的列名中，当前指定的{tokenized_column_name=}，而输入文件的列名为{input_file.columns=}")
    input_file=input_file[tokenized_column_name]
    text_features=vectorizer.fit_transform(input_file)
    ergodicRange=list(range(start_num_topic,end_num_topic+1,num_topic_step))
    if finished_num_topics:
        unfinished_num_topics=sorted(tuple(topic_num for topic_num in ergodicRange if topic_num not in zip(*finished_num_topics).__next__()))
    else:
        unfinished_num_topics=ergodicRange
    if unfinished_num_topics:
        n_jobs_beg=int(os.cpu_count()*0.8)
        if n_jobs_beg<2:
            n_jobs_beg=2
    for unfinished_num_topic in unfinished_num_topics:
        perplexity,n_jobs_beg=runLDAwithRetry(runtime_code,unfinished_num_topic,text_features,df_identity_code,tokenized_column_name,text_feature_extractor,n_jobs_beg,retry=0)
        finished_num_topics.append([unfinished_num_topic,perplexity])
    finished_num_topics.sort(key=lambda x:x[1]) # sort according to the perplexity
    best_num_topic,best_perplexity=finished_num_topics[0]
    lda,lda_dtf=pickle.load(open(f"respawnpoint/{runtime_code}_lda_{best_num_topic}_{best_perplexity}_{df_identity_code}_{tokenized_column_name}_{text_feature_extractor}.pkl","rb"))
    # write the most potential 
    most_likely_topic_id=lda_dtf.argmax(axis=1)+1
    input_file=pickle.load(open(f"respawnpoint/{runtime_code}_input_dataframe_backup.pkl","rb"))
    input_file["most_likely_topic_id"]=most_likely_topic_id
    # likelihood=lda_dtf
    # input_file["likelihood"]=likelihood # this two lines are used when fuzzy_topic parameter come into use
    # save the keywords of each topic
    sorting_all=np.argsort(lda.components_)[:,::-1]
    array_all=np.full((1,sorting_all.shape[1]),1)
    array_all=np.concatenate((array_all,sorting_all),axis=0)
    topics_out=pd.DataFrame(np.array([*vectorizer.get_feature_names_out()[array_all[1:best_num_topic+1,:24]]]).T,columns=[f"Topic {topic_name}" for topic_name in range(1,best_num_topic+1)])
    topics_out.to_excel(f"finalresults/{runtime_code}_Top{best_num_topic}Topics.xlsx")
    # save the final DataFrame
    input_file=input_file.set_index(index_name)
    saveConcatedDataAsFinalResult(runtime_code,input_file,output_filename,clear_respawnpoint_upon_conplete)
    return input_file

def runLDAfromDFusingConfigMenu(config_menu):
    # a shortcut to run the code using a config menu to avoid input to many parameters
    runtime_code=config_menu["runtime_code"]
    input_file=config_menu["input_file"]
    tokenized_column_name=config_menu["tokenized_column_name"]
    text_feature_extractor=config_menu.get("text_feature_extractor","bow")
    start_num_topic=config_menu.get("start_num_topic",4)
    end_num_topic=config_menu.get("end_num_topic",96)
    num_topic_step=config_menu.get("num_topic_step",4)
    output_filename=config_menu.get("output_filename",None)
    clear_respawnpoint_before_run=config_menu.get("clear_respawnpoint_before_run",False)
    clear_respawnpoint_upon_conplete=config_menu.get("clear_respawnpoint_upon_conplete",True)
    return runLDAfromDF(runtime_code,input_file,tokenized_column_name,text_feature_extractor,start_num_topic,end_num_topic,num_topic_step,output_filename,clear_respawnpoint_before_run,clear_respawnpoint_upon_conplete)
import sys
import os
from config.config_loader import load_config
from config.logger import setup_logging
from core.connection import ConnectionHandler
from core.utils.modules_initialize import initialize_modules
import uuid
import asyncio
from core.tts.edge import TTSProvider

async def main():
    try:
        
        config = load_config()
        logger = setup_logging()


        # 初始化模块
        modules = initialize_modules(
            logger,
            config,
            False,
            "ASR" in config["selected_module"],
            "LLM" in config["selected_module"],
            "TTS" in config["selected_module"],
            False,
            False,
        )
        
        # 获取各模块实例
        asr = modules["asr"] if "asr" in modules else None
        llm = modules["llm"] if "llm" in modules else None
        tts = modules["tts"] if "tts" in modules else None

        
        session_id = str(uuid.uuid4())
        
        # 语音识别 - 使用正斜杠避免转义问题
        audio_file_path = r"data\audio.wav"
        logger.info(f"开始处理音频文件: {audio_file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(audio_file_path):
            logger.warning(f"音频文件不存在: {audio_file_path}")
            logger.info("将使用测试文本进行演示")
            text = "你好，这是一个测试。"
            file_path = None
        else:
            # 异步调用ASR方法
            text, file_path = await asr.speech_to_text_from_audio_file(audio_file_path, session_id)
            logger.info(f"语音识别结果: {text}")
            logger.info(f"处理文件路径: {file_path}")
        
        # 创建连接处理器 - 注意参数顺序
        conn = ConnectionHandler(
            config=config,
            _tts=tts,
            _asr=asr,
            _llm=llm
        )
        
        # 进行对话
        if text:
            logger.info(f"开始对话，用户输入: {text}")
            result = conn.chat(text)
            logger.info(f"对话结果: {result}")
            # 生成文件名示例
            file_path= conn.tts.generate_filename()
            await conn.tts.text_to_speak(conn.tts_MessageText, file_path)
            logger.info(f"输出文件: {file_path}")
            # 输出最终回复
            if hasattr(conn, 'tts_MessageText') and conn.tts_MessageText:
                logger.info(f"最终回复: {conn.tts_MessageText}")
        else:
            logger.warning("未识别到有效文本")
            
    except FileNotFoundError as e:
        logger = setup_logging() if 'setup_logging' in globals() else None
        if logger:
            logger.error(f"文件未找到: {e}")
        else:
            print(f"文件未找到: {e}")
    except Exception as e:
        logger = setup_logging() if 'setup_logging' in globals() else None
        if logger:
            logger.error(f"应用执行出错: {e}", exc_info=True)
        else:
            print(f"应用执行出错: {e}")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
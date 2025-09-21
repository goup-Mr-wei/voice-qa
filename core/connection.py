import copy
import json
import uuid
import asyncio
import threading
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from core.utils.dialogue import Message, Dialogue
from core.utils import textUtils
from config.logger import setup_logging

TAG = __name__

class ConnectionHandler:
    def __init__(
        self,
        config: Dict[str, Any],
        _tts,
        _asr,
        _llm,
    ):
        self.config = copy.deepcopy(config)
        self.session_id = str(uuid.uuid4())
        self.logger = setup_logging()

        # 线程任务相关
        # self.loop = asyncio.get_event_loop()
        # self.stop_event = threading.Event()
        # self.executor = ThreadPoolExecutor(max_workers=5)

        # 依赖的组件
        self.vad = None
        self.asr = None
        self.tts = _tts
        self._asr = _asr
        self.llm = _llm

        # llm相关变量
        self.llm_finish_task = True
        self.dialogue = Dialogue()

        # tts相关变量
        self.sentence_id = None
        self.tts_MessageText = ""

        # 初始化提示词管理器
        # self.prompt_manager = PromptManager(config, self.logger)

    def chat(self, query, depth=0):
        self.logger.bind(tag=TAG).info(f"大模型收到用户消息: {query}")
        self.llm_finish_task = False

        # 为最顶层时新建会话ID和发送FIRST请求
        if depth == 0:
            self.sentence_id = str(uuid.uuid4().hex)
            self.dialogue.put(Message(role="user", content=query))

        response_message = []

        try:
            # 调用LLM生成回复
            llm_responses = self.llm.response(
                self.session_id,
                self.dialogue.get_llm_dialogue_with_memory(
                    None, self.config.get("voiceprint", {})  # 不使用memory
                ),
            )
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"LLM 处理出错 {query}: {e}")
            return None

        # 处理流式响应
        self.client_abort = False
        emotion_flag = True
        for content in llm_responses:
            if self.client_abort:
                break

            # 在llm回复中获取情绪表情，一轮对话只在开头获取一次
            if emotion_flag and content is not None and content.strip():
                emotion_flag = False
            if content is not None and len(content) > 0:
                response_message.append(content)

        # 存储对话内容
        if len(response_message) > 0:
            text_buff = "".join(response_message)
            self.tts_MessageText = text_buff
            self.dialogue.put(Message(role="assistant", content=text_buff))
              
        self.llm_finish_task = True
        self.logger.bind(tag=TAG).debug(
            lambda: json.dumps(
                self.dialogue.get_llm_dialogue(), indent=4, ensure_ascii=False
            )
        )

        return True

    async def close(self, ws=None):
        """资源清理方法"""
        try:
            # 触发停止事件
            if self.stop_event:
                self.stop_event.set()

            if self.tts:
                await self.tts.close()

            # 最后关闭线程池
            if self.executor:
                try:
                    self.executor.shutdown(wait=False)
                except Exception as executor_error:
                    self.logger.bind(tag=TAG).error(
                        f"关闭线程池时出错: {executor_error}"
                    )
                self.executor = None

            self.logger.bind(tag=TAG).info("连接资源已释放")
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"关闭连接时出错: {e}")
        finally:
            if self.stop_event:
                self.stop_event.set()
"""
简化版SQL Agent - 移除check_query和rewrite_query，仅保留write_query用于强化学习训练

主要修改:
1. 移除 check_query 和 rewrite_query 节点
2. 简化 graph 结构: write_query -> execute_query -> END
3. 移除相关的 prompt 模板和状态字段
4. 保持其他功能不变
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Literal, Optional, cast
from dataclasses import dataclass

import dotenv
import termcolor
import pymysql
import pandas as pd
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
import signal
import logging
from contextlib import contextmanager
from spider_eval.exce_eval_sql_llm import evaluate_with_llm_reward, LLMRewardModel
from datetime import datetime

import agentlightning as agl

agl.setup_logging(apply_to=[__name__])

logger = logging.getLogger(__name__)


LLM_API_CONFIG = {
    'api_base': 'http://10.250.2.25:8004/v1',
    'api_key': 'EMPTY',
    'model': 'qwen3-235b',
    'temperature': 0.1,
    'max_tokens': 30000
}
llm_model = LLMRewardModel(**LLM_API_CONFIG)


class TimeoutError(Exception):
    pass


@contextmanager
def timeout_handler(seconds: int):
    """使用signal实现超时控制 (仅Unix系统)"""
    def timeout_signal_handler(signum, frame):
        raise TimeoutError(f"Query execution timeout after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@dataclass
class TrainingExample:
    """Training example from MySQL training data"""
    question: str
    query: str
    database_name: str
    table_schema: str
    execution_result: str


SCHEMA = '''
pingshan_stat_info 主要字段:

【基础信息】
- TASK_NUM VARCHAR(40) - 任务编号
- CREATE_TIME TIMESTAMP - 创建时间（默认时间字段）
- OCCUR_TIME TIMESTAMP - 发生时间
- ADDRESS VARCHAR(1000) - 地址

【地理位置】
- STREET_NAME VARCHAR(40) - 街道名称
- COMMUNITY_NAME VARCHAR(40) - 社区名称
- CELL_NAME VARCHAR(40) - 网格名称
- DISTRICT_NAME VARCHAR(40) - 区域名称

【事件分类】
- EVENT_TYPE_NAME VARCHAR(20) - 事件类型（一级）
- MAIN_TYPE_NAME VARCHAR(40) - 事件主类型（二级）
- SUB_TYPE_NAME VARCHAR(100) - 事件子类型（三级）
- THIRD_TYPE_NAME VARCHAR(100) - 事件三级类型（四级）
- MAX_EVENT_TYPE_NAME VARCHAR(200) - 末级分类

【事件来源和渠道】
- EVENT_SRC_NAME VARCHAR(40) - 事件来源（如：i深圳app、微信公众号、机动网格员等）
- REC_TYPE_NAME VARCHAR(40) - 接收类型/上报渠道（如：机动中队、市一体化平台、物联感知等）

【处理单位】
- first_unit_name VARCHAR(200) - 首次处理单位/处置部门
- dispose_unit_name VARCHAR(200) - 处置单位名称
- second_unit_name VARCHAR(200) - 二次处理单位

【标签字段】
- origin_marks VARCHAR(300) - 事件主体标签/主体名称（如：某某学校、花样年华小区等）
- special_marks TEXT - 专项标签
- geography_marks TEXT - 地理标签
- topic_marks TEXT - 话题标签
- hotpoint_marks TEXT - 热点标签
- process_marks TEXT - 流程标签

【事件等级】
- EVENT_GRADE_NAME VARCHAR(20) - 事件等级名称
- event_grade_id INT - 事件等级ID（1=常态事件）

【事件描述】
- EVENT_DESC TEXT - 事件描述

【ID字段（用于特殊查询）】
- event_type_id INT - 事件类型ID
- main_type_id INT - 主类型ID
- sub_type_id INT - 子类型ID
- third_type_id INT - 三级类型ID
- event_src_id INT - 事件来源ID
- rec_type_id INT - 接收类型ID

【其他常用字段】
- archive_time TIMESTAMP - 归档时间
- dispose_begin_time TIMESTAMP - 处置开始时间
- dispose_end_time TIMESTAMP - 处置结束时间
'''


# ========== 仅保留 WRITE_QUERY_PROMPT ==========
WRITE_QUERY_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """你是坪山区城市管理事件数据库的SQL代理。根据问题生成正确的{dialect}查询。

## 核心要求
- 只使用存在的列名
- 生成SELECT查询，不修改数据
- 符合MySQL语法
- 注意区分各类ID字段和NAME字段

## 时间查询标准模板
- **今年**: `WHERE create_time >= DATE_FORMAT(CURDATE(), '%Y-01-01') AND create_time < DATE_FORMAT(CURDATE() + INTERVAL 1 YEAR, '%Y-01-01')`
- **去年**: `WHERE create_time >= DATE_FORMAT(CURDATE() - INTERVAL 1 YEAR, '%Y-01-01') AND create_time < DATE_FORMAT(CURDATE(), '%Y-%m-01')`
- **本月**: `WHERE create_time >= DATE_FORMAT(CURDATE(), '%Y-%m-01') AND create_time < DATE_FORMAT(CURDATE() + INTERVAL 1 MONTH, '%Y-%m-01')`
- **上个月**: `WHERE create_time >= DATE_FORMAT(CURDATE() - INTERVAL 1 MONTH, '%Y-%m-01') AND create_time < DATE_FORMAT(CURDATE(), '%Y-%m-01')`
- **本周**(周一起始): `WHERE create_time >= CURDATE() - INTERVAL (DAYOFWEEK(CURDATE()) - 2) DAY`
- **上周**: `WHERE create_time >= CURDATE() - INTERVAL (DAYOFWEEK(CURDATE()) + 5) DAY AND create_time < CURDATE() - INTERVAL (DAYOFWEEK(CURDATE()) - 2) DAY`
- **最近N天**: `WHERE create_time >= CURDATE() - INTERVAL (N-1) DAY`
- **昨天**: `WHERE create_time >= CURDATE() - INTERVAL 1 DAY AND create_time < CURDATE()`
- **今天**: `WHERE create_time >= CURDATE()`
- **具体日期**: `WHERE DATE_FORMAT(CREATE_TIME, 'yyyy-mm-dd') = '2025-03-04'`
- **年月**: `WHERE DATE_FORMAT(CREATE_TIME, 'yyyy-mm') = '2025-03'`

## 关键业务术语与字段映射

### 时间字段
- **创建时间**（默认） → CREATE_TIME
- **发生时间** → OCCUR_TIME

### 事件分类
- **一级分类/事件类型** → EVENT_TYPE_NAME
- **二级分类/事件主类型** → MAIN_TYPE_NAME
- **三级分类/事件子类型** → SUB_TYPE_NAME
- **四级分类** → THIRD_TYPE_NAME
- **末级分类** → MAX_EVENT_TYPE_NAME

### 来源和渠道（重要区分）
- **事件来源**（谁报告的） → EVENT_SRC_NAME
  * 如：i深圳app、微信公众号、机动网格员、民生诉求等
- **上报渠道/接收类型**（通过什么渠道） → REC_TYPE_NAME
  * 如：机动中队、市一体化平台、物联感知等

### 处理相关
- **处理单位/首次处理单位/处置部门** → first_unit_name
- **办理/处置单位** → dispose_unit_name

### 标签字段（TEXT类型，需要CAST）
- **事件主体/主体标签/主体名称** → origin_marks
  * 查询小区、学校等主体时使用
  * 示例：`origin_marks LIKE '%花样年华%'`
- **专项标签** → special_marks
- **地理标签** → geography_marks
- **话题标签** → topic_marks
- **热点标签** → hotpoint_marks
- **流程标签** → process_marks
  * TEXT字段分组时需要：`GROUP BY CAST(special_marks AS CHAR)`

### 地理位置
- **街道** → STREET_NAME
- **社区** → COMMUNITY_NAME
- **地址** → ADDRESS

### 事件等级
- **常态事件** → event_grade_id = 1

### 关键词匹配技巧
- 查询"暴露垃圾"等关键词时，需要多字段LIKE匹配：
  ```sql
  WHERE (SUB_TYPE_NAME LIKE '%暴露垃圾%' 
         OR THIRD_TYPE_NAME LIKE '%暴露垃圾%' 
         OR MAX_EVENT_TYPE_NAME LIKE '%暴露垃圾%'
         OR EVENT_DESC LIKE '%暴露垃圾%')
  ```

## 常见查询模板

### 1. 统计某类事件数量
```sql
SELECT COUNT(*) AS 事件数量
FROM pingshan_stat_info
WHERE SUB_TYPE_NAME LIKE '%关键词%'
  AND create_time >= [时间条件]
```

### 2. 按维度分组统计
```sql
SELECT STREET_NAME AS 街道, COUNT(*) AS 数量
FROM pingshan_stat_info
WHERE [条件]
GROUP BY STREET_NAME
ORDER BY 数量 DESC
```

### 3. 主体相关查询
```sql
SELECT origin_marks AS 主体, COUNT(*) AS 数量
FROM pingshan_stat_info
WHERE (origin_marks LIKE '%主体名%' 
       OR address LIKE '%主体名%'
       OR event_desc LIKE '%主体名%')
  AND [时间条件]
GROUP BY origin_marks
```

### 4. TEXT字段分组（需要CAST）
```sql
SELECT CAST(special_marks AS CHAR) AS 专项类型, COUNT(*) AS 数量
FROM pingshan_stat_info
WHERE special_marks IS NOT NULL 
  AND special_marks != ''
GROUP BY CAST(special_marks AS CHAR)
ORDER BY 数量 DESC
```

## 注意事项
1. **时间字段默认用CREATE_TIME**，除非明确要求发生时间
2. **区分event_src_name和rec_type_name**：来源vs渠道
3. **TEXT字段分组必须CAST**：`CAST(xxx AS CHAR)`
4. **关键词匹配多字段**：SUB_TYPE_NAME、THIRD_TYPE_NAME、MAX_EVENT_TYPE_NAME、EVENT_DESC
5. **主体查询三字段**：origin_marks、address、event_desc
6. **非空判断**：`IS NOT NULL AND != ''`

## 表结构
{table_info}

## 输出格式
```{dialect}
YOUR SQL QUERY HERE
```
""".strip(),
        ),
        ("user", "问题: {input}"),
    ]
)


# ========== 简化的 State 类 - 移除不需要的字段 ==========
class State(MessagesState):
    question: str
    query: str
    execution: str
    messages: list[BaseMessage]


class SQLAgent:

    def __init__(
        self,
        db: str,
        debug: bool = False,
        db_schema: str | None = None,
        endpoint: str | None = None,
        verl_replacement: dict | None = None,
        table_info_truncate: int = 3000,
        execution_truncate: int = 3000,
    ):
        self.db = SQLDatabase.from_uri(db)
        self.db_schema = db_schema
        self.debug = debug
        self.table_info_truncate = table_info_truncate
        self.execution_truncate = execution_truncate
        
        if verl_replacement is not None:
            self.model_name = verl_replacement["model"]
            assert endpoint is not None
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint,
                openai_api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
                temperature=verl_replacement["temperature"],
                max_retries=0,
                max_tokens=2048,
            )
        else:
            self.model_name = os.environ.get("MODEL", "qwen3-code")
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base="http://10.250.2.24:8005/v1",
                openai_api_key="none",
                temperature=0,
                max_retries=3,
                max_tokens=2048,
            )

    def get_table_info(self) -> str:
        """返回数据库表结构信息"""
        return SCHEMA

    def invoke_prompt(self, prompt: Any, max_retries: int = 3) -> BaseMessage:
        """带重试和验证的提示词调用"""
        if self.debug:
            for message in prompt.messages:
                termcolor.cprint(message.pretty_repr(), "blue")

        for attempt in range(max_retries):
            try:
                result = self.llm.invoke(prompt)
                
                if not result or not result.content:
                    logger.warning(f"Attempt {attempt + 1}: Empty response received")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        logger.error("All retries exhausted, returning default query")
                        return BaseMessage(
                            content="```mysql\nSELECT COUNT(*) FROM pingshan_stat_info;\n```",
                            type="ai"
                        )
                
                if self.debug:
                    termcolor.cprint(result.pretty_repr(), "green")
                
                return result
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    return BaseMessage(
                        content="```mysql\nSELECT COUNT(*) FROM pingshan_stat_info;\n```",
                        type="ai"
                    )
        
        return BaseMessage(
            content="```mysql\nSELECT COUNT(*) FROM pingshan_stat_info;\n```",
            type="ai"
        )

    def parse_query(self, message: BaseMessage) -> str | None:
        """从消息中解析SQL查询"""
        result = None
        for match in re.finditer(r".*```\w*\n(.*?)\n```.*", message.content, re.DOTALL):
            result = match.group(1).strip()
        return result

    def write_query(self, state: State):
        """生成SQL查询 - 唯一的查询生成步骤"""
        try:
            prompt = WRITE_QUERY_PROMPT.invoke(
                {
                    "dialect": self.db.dialect,
                    "input": state["question"][:500],
                    "table_info": self.get_table_info(),
                }
            )
            result = self.invoke_prompt(prompt, max_retries=3)

            query = self.parse_query(result)
            if not query:
                query = result.content
                query = query.replace("```mysql", "").replace("```sql", "").replace("```", "").strip()
            
            if not query or len(query) < 10:
                logger.error(f"Invalid query generated: {query}")
                query = "SELECT COUNT(*) FROM pingshan_stat_info;"
            
            return {
                **state,
                "query": query,
                "messages": [*prompt.messages, result],
            }
            
        except Exception as e:
            logger.error(f"Error in write_query: {e}")
            return {
                **state,
                "query": "SELECT COUNT(*) FROM pingshan_stat_info;",
                "messages": [],
            }

    def execute_query(self, state: State, timeout_seconds: int = 100) -> State:
        """执行SQL查询"""
        execute_query_tool = QuerySQLDatabaseTool(db=self.db)
        try:
            with timeout_handler(timeout_seconds):
                execution_result = execute_query_tool.invoke(state["query"])
                if not isinstance(execution_result, str):
                    execution_result = str(execution_result)
                if self.debug:
                    termcolor.cprint(execution_result, "yellow")
                return {**state, "execution": execution_result}
        except TimeoutError as e:
            error_msg = str(e)
            if self.debug:
                termcolor.cprint(error_msg, "red")
            return {**state, "execution": error_msg}

    # ========== 移除 check_query 和 rewrite_query 方法 ==========

    def graph(self) -> CompiledStateGraph[State]:
        """构建简化的图结构: write_query -> execute_query -> END"""
        builder = StateGraph(State)
        
        # 只添加两个节点
        builder.add_node(self.write_query)
        builder.add_node(self.execute_query)

        # 简化的边连接
        builder.add_edge(START, "write_query")
        builder.add_edge("write_query", "execute_query")
        builder.add_edge("execute_query", END)  # 直接结束，不再检查和重写

        return builder.compile()


class LitSQLAgent(agl.LitAgent):

    def __init__(
        self,
        trained_agents: Optional[str] = r"write",
        val_temperature: Optional[float] = None,
        table_info_truncate: int = 2000,
        execution_truncate: int = 1500,
    ) -> None:
        super().__init__(trained_agents=trained_agents)
        self.val_temperature = val_temperature
        self.table_info_truncate = table_info_truncate
        self.execution_truncate = execution_truncate
        
        # MySQL configuration
        self.mysql_config = {
            'host': os.environ.get('MYSQL_HOST', '10.250.2.19'),
            'port': int(os.environ.get('MYSQL_PORT', 3306)),
            'user': os.environ.get('MYSQL_USER', 'root'),
            'password': os.environ.get('MYSQL_PASSWORD', 'hwits888'),
            'database': os.environ.get('MYSQL_DATABASE', 'pingshan'),
        }

    def rollout(
        self,
        task,
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        # Handle both dict and TrainingExample formats
        if isinstance(task, dict):
            question = task["question"]
            ground_truth = task["query"]
            db_schema = SCHEMA[:2000]
        else:
            question = task.question
            ground_truth = task.query
            db_schema = SCHEMA[:2000]
            
        start_time = time.time()
        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])

        # Create MySQL connection URI
        mysql_uri = f"mysql+pymysql://{self.mysql_config['user']}:{self.mysql_config['password']}@{self.mysql_config['host']}:{self.mysql_config['port']}/{self.mysql_config['database']}"
        
        rollout_id = rollout.rollout_id
        
        logger.info(f"[Rollout {rollout_id}] Question: {question}")
        logger.info(f"[Rollout {rollout_id}] Ground Truth: {ground_truth}")

        is_training = (rollout.mode == "train")

        # 创建简化的 agent
        agent = SQLAgent(
            mysql_uri,
            table_info_truncate=2000,
            execution_truncate=1500,
            debug=False,
            db_schema=db_schema,
            endpoint=llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id),
            verl_replacement=(
                {"model": llm.model, **llm.sampling_parameters}
                if is_training
                else {
                    "model": llm.model,
                    "temperature": (
                        self.val_temperature
                        if self.val_temperature is not None
                        else llm.sampling_parameters.get("temperature", 0.0)
                    ),
                }
            ),
        ).graph()
        
        try:
            handler = self.tracer.get_langchain_handler()
            result = agent.invoke(
                {"question": question},
                {"callbacks": [handler] if handler else [], "recursion_limit": 100},
            )
        except Exception as e:
            logger.exception(f"[Rollout {rollout_id}] Error during agent invocation: {e}")
            return 0.0

        # 验证生成结果
        if not result or 'query' not in result or not result['query']:
            logger.error(f"[Rollout {rollout_id}] Invalid result")
            return 0.0
        
        generated_query = result['query']
        
        if not isinstance(generated_query, str) or len(generated_query.strip()) < 10:
            logger.error(f"[Rollout {rollout_id}] Invalid generated query")
            return 0.0

        logger.info(f"[Rollout {rollout_id}] Generated Query: {generated_query}")

        end_time_rollout = time.time()

        # 评估奖励
        reward, details = evaluate_with_llm_reward(
            question=question,
            predicted_query=generated_query,
            ground_truth_query=ground_truth,
            mysql_config=self.mysql_config,
            llm_reward_model=llm_model,
            llm_weight=1,
            execution_weight=0,
            return_details=True
        )

        logger.info("[Rollout %s] Reward: %s", rollout_id, reward)

        end_time_eval = time.time()
        logger.info("[Rollout %s] Time taken for rollout: %.2f seconds", rollout_id, end_time_rollout - start_time)
        logger.info("[Rollout %s] Time taken for evaluation: %.2f seconds", rollout_id, end_time_eval - end_time_rollout)

        return reward


def debug_sql_agent():
    """调试函数"""
    spider_dev_data_path = os.path.join(
        os.environ.get("VERL_SPIDER_DATA_DIR", "/home/user/cx/new_data/agent-lightning/examples/spider/data_claude/"),
        "test_data.parquet"
    )
    if not os.path.exists(spider_dev_data_path):
        raise FileNotFoundError(f"Spider dev data file {spider_dev_data_path} does not exist.")
    
    df = pd.read_parquet(spider_dev_data_path).head(10)
    df = cast(List[Dict[str, Any]], df.to_dict(orient="records"))
    print("Debug data:", df)

    trainer = agl.Trainer(
        n_workers=1,
        initial_resources={
            "main_llm": agl.LLM(
                endpoint=os.environ["OPENAI_API_BASE"],
                model="qwen3-235b",
                sampling_parameters={"temperature": 0.7},
            )
        },
    )
    trainer.dev(LitSQLAgent(), df)


def test_sql_agent_from_excel(
    excel_path: str,
    output_path: str = None,
    question_col: str = "question",
    sql_col: str = "query",
    temperature: float = 0.0,
):
    """
    从Excel文件加载测试数据并评估简化版agent
    """
    logger.info(f"Loading test data from {excel_path}")
    try:
        df = pd.read_excel(excel_path)
        logger.info(f"Loaded {len(df)} test cases")
    except Exception as e:
        logger.error(f"Failed to load Excel file: {e}")
        raise
    
    if question_col not in df.columns or sql_col not in df.columns:
        raise ValueError(f"Required columns not found in Excel file")
    
    if output_path is None:
        base_name = os.path.splitext(excel_path)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{base_name}_result_{timestamp}.xlsx"
    
    mysql_config = {
        'host': os.environ.get('MYSQL_HOST', '10.250.2.19'),
        'port': int(os.environ.get('MYSQL_PORT', 3306)),
        'user': os.environ.get('MYSQL_USER', 'root'),
        'password': os.environ.get('MYSQL_PASSWORD', 'hwits888'),
        'database': os.environ.get('MYSQL_DATABASE', 'pingshan'),
    }
    
    mysql_uri = f"mysql+pymysql://{mysql_config['user']}:{mysql_config['password']}@{mysql_config['host']}:{mysql_config['port']}/{mysql_config['database']}"
    
    results = {
        'question': [],
        'ground_truth_sql': [],
        'generated_sql': [],
        'reward': [],
        'execution_status': [],
        'error_message': []
    }
    
    for idx, row in df.iterrows():
        question = row[question_col]
        ground_truth = row[sql_col]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Test Case {idx + 1}/{len(df)}")
        logger.info(f"Question: {question}")
        
        results['question'].append(question)
        results['ground_truth_sql'].append(ground_truth)
        
        try:
            agent = SQLAgent(
                mysql_uri,
                table_info_truncate=3000,
                execution_truncate=3000,
                debug=False,
                db_schema=SCHEMA
            ).graph()
            
            result = agent.invoke(
                {"question": question},
                {"recursion_limit": 10}
            )
            
            generated_sql = result.get('query', '')
            logger.info(f"Generated SQL: {generated_sql}")
            
            reward, details = evaluate_with_llm_reward(
                question=question,
                predicted_query=generated_sql,
                ground_truth_query=ground_truth,
                mysql_config=mysql_config,
                llm_reward_model=llm_model,
                llm_weight=1,
                execution_weight=0,
                return_details=True
            )
            
            logger.info(f"Reward: {reward}")
            
            results['generated_sql'].append(generated_sql)
            results['reward'].append(reward)
            results['execution_status'].append('SUCCESS')
            results['error_message'].append('')
            
        except Exception as e:
            logger.error(f"Error processing test case {idx + 1}: {e}")
            results['generated_sql'].append('')
            results['reward'].append(0.0)
            results['execution_status'].append('FAILED')
            results['error_message'].append(str(e))
    
    result_df = pd.DataFrame(results)
    
    total_cases = len(result_df)
    successful_cases = len(result_df[result_df['execution_status'] == 'SUCCESS'])
    avg_reward = result_df['reward'].mean()
    perfect_matches = len(result_df[result_df['reward'] == 1.0])
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Test Summary:")
    logger.info(f"Total test cases: {total_cases}")
    logger.info(f"Successful executions: {successful_cases} ({successful_cases/total_cases*100:.2f}%)")
    logger.info(f"Perfect matches (reward=1.0): {perfect_matches} ({perfect_matches/total_cases*100:.2f}%)")
    logger.info(f"Average reward: {avg_reward:.4f}")
    
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            result_df.to_excel(writer, sheet_name='Results', index=False)
            
            summary_df = pd.DataFrame({
                'Metric': [
                    'Total Cases',
                    'Successful Executions',
                    'Success Rate (%)',
                    'Perfect Matches',
                    'Perfect Match Rate (%)',
                    'Average Reward',
                    'Test Time'
                ],
                'Value': [
                    total_cases,
                    successful_cases,
                    f"{successful_cases/total_cases*100:.2f}",
                    perfect_matches,
                    f"{perfect_matches/total_cases*100:.2f}",
                    f"{avg_reward:.4f}",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise
    
    return result_df


def debug_test_from_excel():
    """调试函数示例"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    excel_path = "/home/user/cx/new_data/agent-lightning/examples/spider/坪山测试数据原始sql.xlsx"
    output_path = "/home/user/cx/new_data/agent-lightning/examples/spider/坪山测试数据_simplified_results.xlsx"
    
    result_df = test_sql_agent_from_excel(
        excel_path=excel_path,
        output_path=output_path,
        question_col="question",
        sql_col="query",
        temperature=0.0
    )
    
    print("\nFirst 5 results:")
    print(result_df.head())
    
    return result_df


if __name__ == "__main__":
    debug_sql_agent()
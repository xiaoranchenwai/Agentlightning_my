import asyncio
import re
from typing import Any, List, Set, Tuple, Dict, Optional
from collections import defaultdict
from itertools import chain, product
import random
import pymysql
import tqdm
import logging
import json

TIMEOUT = 30

logger = logging.getLogger(__name__)


# ==================== 原有的SQL执行和比较函数 ====================

def permute_tuple(element: Tuple, perm: Tuple) -> Tuple:
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])


def unorder_row(row: Tuple) -> Tuple:
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


def quick_rej(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        return s1 == s2
    else:
        return set(s1) == set(s2)


def multiset_eq(l1: List, l2: List) -> bool:
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False
    return True


def get_constraint_permutation(tab1_sets_by_columns: List[Set], result2: List[Tuple]):
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)

    for _ in range(20):
        random_tab2_row = random.choice(result2)
        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)
    return product(*perm_constraints)


def result_eq(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    """Check whether two denotations are equivalent"""
    if len(result1) == 0 and len(result2) == 0:
        return True

    if len(result1) != len(result2):
        return False

    num_cols = len(result1[0])

    if len(result2[0]) != num_cols:
        return False

    if not quick_rej(result1, result2, order_matters):
        return False

    tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]

    for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
        if len(perm) != len(set(perm)):
            continue
        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]
        if order_matters:
            if result1 == result2_perm:
                return True
        else:
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                return True
    return False


def calculate_result_similarity(result1: List[Tuple], result2: List[Tuple]) -> float:
    """计算两个查询结果的相似度"""
    if len(result1) == 0 and len(result2) == 0:
        return 1.0
    
    if len(result1) == 0 or len(result2) == 0:
        return 0.0
    
    if len(result1[0]) != len(result2[0]):
        return 0.1
    
    row_similarity = 1.0 - abs(len(result1) - len(result2)) / max(len(result1), len(result2))
    
    set1 = set(unorder_row(row) for row in result1)
    set2 = set(unorder_row(row) for row in result2)
    
    if len(set1) == 0 and len(set2) == 0:
        overlap = 1.0
    else:
        overlap = len(set1 & set2) / max(len(set1), len(set2))
    
    similarity = 0.3 * row_similarity + 0.7 * overlap
    
    return similarity


def replace_cur_year(query: str) -> str:
    return re.sub(r"YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)", "2020", query, flags=re.IGNORECASE)


def postprocess(query: str) -> str:
    query = query.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
    return query


def remove_distinct(query: str) -> str:
    return re.sub(r'\bDISTINCT\b', '', query, flags=re.IGNORECASE)


def analyze_sql_syntax(query: str) -> Dict[str, float]:
    """分析SQL语法特征"""
    rewards = {}
    query_lower = query.lower()
    
    has_select = 'select' in query_lower
    has_from = 'from' in query_lower
    rewards['basic_syntax'] = 0.2 if (has_select and has_from) else 0.0
    
    keywords = ['where', 'group by', 'order by', 'having', 'join', 'limit']
    rewards['has_keywords'] = sum(0.05 for kw in keywords if kw in query_lower)
    
    open_parens = query.count('(')
    close_parens = query.count(')')
    rewards['balanced_parens'] = 0.1 if open_parens == close_parens else 0.0
    
    return rewards


def exec_on_mysql_sync(mysql_config: dict, query: str, timeout: int = TIMEOUT) -> Tuple[str, Any]:
    """Execute query on MySQL database synchronously with timeout"""
    import signal
    
    query = replace_cur_year(query)
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Query execution timeout")
    
    connection = None
    cursor = None
    
    try:
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        
        connection = pymysql.connect(**mysql_config, connect_timeout=timeout)
        cursor = connection.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        
        return "result", result
        
    except TimeoutError:
        return "timeout", None
    except pymysql.err.ProgrammingError as e:
        return "syntax_error", str(e)
    except pymysql.err.OperationalError as e:
        return "operational_error", str(e)
    except Exception as e:
        return "exception", str(e)
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        
        if cursor:
            cursor.close()
        if connection:
            connection.close()


# ==================== LLM奖励模型 ====================

class LLMRewardModel:
    """使用LLM作为奖励模型来评估SQL质量"""
    
    def __init__(
        self, 
        api_base: str,
        api_key: str = "EMPTY",
        model: str = "default",
        temperature: float = 0.1,
        max_tokens: int = 512
    ):
        """
        初始化LLM奖励模型
        
        Args:
            api_base: OpenAI格式API的base URL，例如 "http://localhost:8000/v1"
            api_key: API密钥，本地部署通常用 "EMPTY"
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成token数
        """
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 检查是否安装了openai库
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=api_base
            )
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def _build_evaluation_prompt(
        self,
        question: str,
        predicted_sql: str,
        ground_truth_sql: str,
        predicted_result: Optional[Any] = None,
        ground_truth_result: Optional[Any] = None,
        execution_status: str = "unknown",
        error_message: Optional[str] = None
    ) -> str:
        """构建评估prompt"""
        
        prompt = f"""你是一个SQL查询质量评估专家。请评估预测的SQL查询的质量。

**用户问题（中文）：**
{question}

**标准SQL查询：**
```sql
{ground_truth_sql}
```

**预测SQL查询：**
```sql
{predicted_sql}
```

**执行状态：** {execution_status}
"""
        
        if error_message:
            prompt += f"\n**错误信息：** {error_message}\n"
        
        if predicted_result is not None and ground_truth_result is not None:
            prompt += f"""
**标准查询结果（前5行）：**
{self._format_result(ground_truth_result, max_rows=5)}

**预测查询结果（前5行）：**
{self._format_result(predicted_result, max_rows=5)}
"""
        
        prompt += """
请从以下维度评估预测SQL的质量，并给出0-100的综合分数：

1. **语义正确性（40分）**：预测SQL是否正确理解了用户问题的意图
2. **语法正确性（20分）**：SQL语法是否正确，能否成功执行
3. **结果准确性（30分）**：查询结果是否与标准答案一致或相近
4. **查询效率（10分）**：SQL是否高效，是否有不必要的复杂操作

请以JSON格式返回评估结果：
```json
{
    "semantic_correctness": <0-40>,
    "syntax_correctness": <0-20>,
    "result_accuracy": <0-30>,
    "query_efficiency": <0-10>,
    "total_score": <0-100>,
    "reasoning": "<简短的评估理由>"
}
```

只返回JSON，不要其他内容。"""
        
        return prompt
    
    def _format_result(self, result: List[Tuple], max_rows: int = 5) -> str:
        """格式化查询结果用于显示"""
        if not result:
            return "空结果"
        
        formatted = []
        for i, row in enumerate(result[:max_rows]):
            formatted.append(str(row))
        
        if len(result) > max_rows:
            formatted.append(f"... (共{len(result)}行)")
        
        return "\n".join(formatted)
    
    def evaluate(
        self,
        question: str,
        predicted_sql: str,
        ground_truth_sql: str,
        predicted_result: Optional[Any] = None,
        ground_truth_result: Optional[Any] = None,
        execution_status: str = "unknown",
        error_message: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        使用LLM评估SQL查询质量
        
        Returns:
            包含评估结果的字典，包括各维度分数和总分
        """
        prompt = self._build_evaluation_prompt(
            question=question,
            predicted_sql=predicted_sql,
            ground_truth_sql=ground_truth_sql,
            predicted_result=predicted_result,
            ground_truth_result=ground_truth_result,
            execution_status=execution_status,
            error_message=error_message
        )
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个SQL查询质量评估专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                content = response.choices[0].message.content.strip()
                
                # 提取JSON
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    
                    # 验证必需字段
                    required_fields = ['semantic_correctness', 'syntax_correctness', 
                                     'result_accuracy', 'query_efficiency', 'total_score']
                    if all(field in result for field in required_fields):
                        return result
                
                logger.warning(f"LLM返回格式不正确，尝试 {attempt + 1}/{max_retries}")
                
            except Exception as e:
                logger.error(f"LLM评估出错 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    # 返回默认低分
                    return {
                        'semantic_correctness': 0,
                        'syntax_correctness': 0,
                        'result_accuracy': 0,
                        'query_efficiency': 0,
                        'total_score': 0,
                        'reasoning': f"LLM评估失败: {str(e)}"
                    }
        
        # 如果所有重试都失败，返回默认值
        return {
            'semantic_correctness': 0,
            'syntax_correctness': 0,
            'result_accuracy': 0,
            'query_efficiency': 0,
            'total_score': 0,
            'reasoning': "LLM评估失败"
        }
    
    def evaluate_batch(
        self,
        questions: List[str],
        predicted_sqls: List[str],
        ground_truth_sqls: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """批量评估"""
        results = []
        
        iterator = zip(questions, predicted_sqls, ground_truth_sqls)
        if show_progress:
            iterator = tqdm.tqdm(list(iterator), desc="LLM评估中")
        
        for question, pred_sql, gold_sql in iterator:
            result = self.evaluate(
                question=question,
                predicted_sql=pred_sql,
                ground_truth_sql=gold_sql
            )
            results.append(result)
        
        return results


# ==================== 混合评估函数 ====================

def evaluate_with_llm_reward(
    question: str,
    predicted_query: str,
    ground_truth_query: str,
    mysql_config: dict,
    llm_reward_model: Optional[LLMRewardModel] = None,
    llm_weight: float = 0.5,
    execution_weight: float = 0.5,
    keep_distinct: bool = False,
    return_details: bool = False
) -> float:
    """
    结合执行结果和LLM判断的混合评估
    
    Args:
        question: 用户的自然语言问题
        predicted_query: 预测的SQL查询
        ground_truth_query: 真实的SQL查询
        mysql_config: MySQL连接配置
        llm_reward_model: LLM奖励模型实例
        llm_weight: LLM评估的权重 (0-1)
        execution_weight: 执行结果的权重 (0-1)
        keep_distinct: 是否保留DISTINCT
        return_details: 是否返回详细信息
    
    Returns:
        reward: 奖励值 [0.0, 1.0]
    """
    details = {
        'execution_reward': 0.0,
        'llm_reward': 0.0,
        'final_reward': 0.0,
        'execution_details': {},
        'llm_details': {}
    }
    
    # Postprocess queries
    p_str = postprocess(predicted_query)
    g_str = postprocess(ground_truth_query)
    
    if not keep_distinct:
        p_str = remove_distinct(p_str)
        g_str = remove_distinct(g_str)
    
    order_matters = "order by" in g_str.lower()
    
    # ===== 1. 执行结果评估 =====
    #if execution_weight!=0:
    try:
        g_flag, g_result = exec_on_mysql_sync(mysql_config, g_str)
        
        if g_flag != "result":
            logger.warning(f"Gold query execution failed: {g_flag}")
            return (0.0, details) if return_details else 0.0
        
        p_flag, p_result = exec_on_mysql_sync(mysql_config, p_str)
        
        # 计算执行奖励
        syntax_rewards = analyze_sql_syntax(p_str)
        base_syntax_reward = sum(syntax_rewards.values())
        
        if p_flag == "result":
            if result_eq(g_result, p_result, order_matters=order_matters):
                execution_reward = 1.0
            else:
                similarity = calculate_result_similarity(g_result, p_result)
                execution_reward = 0.3 * base_syntax_reward + 0.7 * similarity
        elif p_flag == "timeout":
            execution_reward = 0.1 * base_syntax_reward
        elif p_flag == "syntax_error":
            execution_reward = 0.05 * base_syntax_reward
        else:
            execution_reward = 0.02 * base_syntax_reward
        
        details['execution_reward'] = execution_reward
        details['execution_details'] = {
            'status': p_flag,
            'syntax_rewards': syntax_rewards,
            'result_match': 1.0 if (p_flag == "result" and result_eq(g_result, p_result, order_matters)) else 0.0
        }
        
    except Exception as e:
        logger.error(f"Execution evaluation error: {e}")
        execution_reward = 0.0
        p_flag = "exception"
        p_result = None
        g_result = None
    
    # ===== 2. LLM评估 =====
    if llm_reward_model is not None:
        try:
            llm_eval = llm_reward_model.evaluate(
                question=question,
                predicted_sql=p_str,
                ground_truth_sql=g_str,
                predicted_result=p_result if p_flag == "result" else None,
                ground_truth_result=g_result if g_flag == "result" else None,
                execution_status=p_flag,
                error_message=str(p_result) if p_flag != "result" else None
            )
            
            # 将LLM的0-100分数转换为0-1
            llm_reward = llm_eval['total_score'] / 100.0
            details['llm_reward'] = llm_reward
            details['llm_details'] = llm_eval
            
        except Exception as e:
            logger.error(f"LLM evaluation error: {e}")
            llm_reward = execution_reward  # 失败时使用执行奖励
            details['llm_details'] = {'error': str(e)}
    else:
        llm_reward = execution_reward  # 没有LLM模型时使用执行奖励
    
    # ===== 3. 综合奖励 =====
    # 确保权重总和为1
    total_weight = llm_weight + execution_weight
    llm_weight = llm_weight / total_weight
    execution_weight = execution_weight / total_weight
    
    final_reward = llm_weight * llm_reward + execution_weight * execution_reward
    details['final_reward'] = final_reward
    details['weights'] = {'llm': llm_weight, 'execution': execution_weight}
    
    return (final_reward, details) if return_details else final_reward


def evaluate_batch_with_llm(
    questions: List[str],
    predicted_queries: List[str],
    ground_truth_queries: List[str],
    mysql_config: dict,
    llm_reward_model: Optional[LLMRewardModel] = None,
    llm_weight: float = 0.5,
    execution_weight: float = 0.5,
    keep_distinct: bool = False,
    show_progress: bool = True
) -> List[float]:
    """
    批量评估（混合模式）
    """
    assert len(questions) == len(predicted_queries) == len(ground_truth_queries), \
        "All input lists must have the same length"
    
    rewards = []
    iterator = zip(questions, predicted_queries, ground_truth_queries)
    
    if show_progress:
        iterator = tqdm.tqdm(list(iterator), desc="混合评估中")
    
    for question, pred_query, gold_query in iterator:
        reward = evaluate_with_llm_reward(
            question=question,
            predicted_query=pred_query,
            ground_truth_query=gold_query,
            mysql_config=mysql_config,
            llm_reward_model=llm_reward_model,
            llm_weight=llm_weight,
            execution_weight=execution_weight,
            keep_distinct=keep_distinct
        )
        rewards.append(reward)
    
    return rewards


# ==================== 向后兼容 ====================

def evaluate_mysql_query_for_rl(
    predicted_query: str,
    ground_truth_query: str,
    mysql_config: dict,
    keep_distinct: bool = False,
    return_details: bool = False
) -> float:
    """向后兼容的评估函数（仅执行评估）"""
    details = {
        'execution_status': None,
        'result_match': 0.0,
        'syntax_rewards': {},
        'error_type': None
    }
    
    p_str = postprocess(predicted_query)
    g_str = postprocess(ground_truth_query)
    
    if not keep_distinct:
        p_str = remove_distinct(p_str)
        g_str = remove_distinct(g_str)
    
    syntax_rewards = analyze_sql_syntax(p_str)
    details['syntax_rewards'] = syntax_rewards
    base_syntax_reward = sum(syntax_rewards.values())
    
    order_matters = "order by" in g_str.lower()
    
    try:
        g_flag, g_result = exec_on_mysql_sync(mysql_config, g_str)
        
        if g_flag != "result":
            logger.warning(f"Gold query execution failed: {g_flag}")
            details['error_type'] = f"gold_{g_flag}"
            return (0.0, details) if return_details else 0.0
        
        details['gold_result_size'] = len(g_result) if g_result else 0
        
        p_flag, p_result = exec_on_mysql_sync(mysql_config, p_str)
        details['execution_status'] = p_flag
        
        if p_flag == "result":
            if result_eq(g_result, p_result, order_matters=order_matters):
                reward = 1.0
                details['result_match'] = 1.0
            else:
                similarity = calculate_result_similarity(g_result, p_result)
                reward = 0.3 * base_syntax_reward + 0.7 * similarity
                details['result_match'] = similarity
        elif p_flag == "timeout":
            reward = 0.1 * base_syntax_reward
            details['error_type'] = 'timeout'
        elif p_flag == "syntax_error":
            reward = 0.05 * base_syntax_reward
            details['error_type'] = 'syntax_error'
        else:
            reward = 0.02 * base_syntax_reward
            details['error_type'] = p_flag
        
        details['final_reward'] = reward
        return (reward, details) if return_details else reward
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        details['error_type'] = f"evaluation_error: {str(e)}"
        return (0.0, details) if return_details else 0.0
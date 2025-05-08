## <b>背景</b>

<p style="text-align:start">根据业务经验，OBS资源中80%的资源使用量来源于top 20%的客户。未来 OBS 资源的主要增长趋势将集中在 NA 客户，因而 NA 客户的资源使用量预测成为业务重点。</p>

* <b>时间序列的复杂性：</b>OBS资源使用量的时间序列既具有稳定的长期趋势，又夹杂短期的波动与变化，导致传统预测模型难以平衡。
* <b>特征提取的重要性：</b>精准刻画客户的使用模式，才能制定有效的资源配置策略并保障业务需求。




<p style="text-align:start">当前已有基于 Prophet 模型的预测方案，但存在如下问题：</p>

* <b>短期趋势变化的预测：</b>近期趋势变化较大的客户，预测未能突出短期趋势特征。
* <b>数据不足时的预测：</b>新上量客户（历史数据小于 150 天）无法获得预测结果。
* <b>差异化趋势建模：</b>不同客户的趋势特征有显著差异，算法未针对锯齿状、缓慢增长等模式进行显性区分。




<p style="text-align:start"> </p>

## <b>预测范围与效果评估</b>

#### <b>预测范围和维度</b>

* <b>客户+集群</b>：预测范围是具体到每个客户和集群。
* <b>预测数据范围</b>：包括已存在于系统中的top NA客户和所有NA客户。
* <b>预测长度</b>：预测未来3个月，按周进行预测。




#### <b>算法上线后效果评估</b>

* <b>算法侧</b>：使用SMAPE和MAE等指标评估预测准确性。
* <b>业务侧</b>：
    * <b>提高人工效率</b>：通过提高预测覆盖率和准确性，减少人工干预。
    * <b>节省成本投入</b>：通过更准确的资源预测，优化资源配置，降低不必要的资源浪费。







<p style="text-align:start"> </p>

## <b>时序预测</b>

<p style="text-align:unset">基于现有的 Prophet 模型框架，优化数据预处理与模型设计以更好地捕捉复杂时间序列趋势。</p>

<p style="text-align:unset"><b>（1）数据预处理与特征提取</b>：</p>

* <b>异常值检测与修复</b>：
    * 使用 Isolation Forest 检测资源使用中的异常值，并通过插值进行修复。



* <b>短期趋势强化</b>：
    * 调整 Prophet 参数（ <code>changepoint_prior_scale</code> 和 <code>seasonality_prior_scale</code>），加强对近期趋势的捕捉能力。







<b>（2）模型改进</b>：

* <b>近期趋势捕捉：</b>
    * 通过动态窗口长度调节来适应不同客户的时间序列特点，将最近数据赋予更高权重，从而优化短期趋势的预测精度。



* <b>新上量客户处理：</b>
    * 使用数据插值与相似客户历史数据参考，为数据不足的客户提供预测解决方案。



* <b>适配复杂特征：</b>
    * 根据客户数据特点动态调整季节性权重、变点灵敏度等模型参数，优化复杂趋势的预测效果。







<b>（3）评估与可视化</b>

* 当前两种优化方案在部分数据上有效果提升，还有进一步提升空间（调参、分类预测等）。




#### <b>算法概览</b>

<p style="text-align:start"><b>1. 原算法的问题</b> </p>

* <b>异常值处理欠缺</b>：对输入数据中的异常值没有处理。
* <b>代码复杂性高</b>：在<code>_data_process</code>方法中，存在多次循环和复杂的数据操作逻辑。
* <b>灵活性不足</b>：<code>predict</code>方法仅支持按周预测，没有提供按天预测的选项。
* <b>未处理缺失值</b>：对时间序列数据中的缺失值没有合理的插值方法。




<p style="text-align:start"> </p>

<p style="text-align:start"><b>2. 优化算法改进点</b></p>

* <b>异常值检测</b>：引入了<code>Isolation Forest</code>模型对数据进行异常值检测，标记并删除异常值。
* <b>灵活性增强</b>：增加了按天预测的能力。
* <b>模型参数调整</b>：对<code>changepoint_prior_scale</code>等参数进行了优化，降低了模型过拟合的可能性。




<p style="text-align:start"> </p>

#### <b>参数调整</b>

* <code>changepoint_prior_scale</code>
    * <b>原算法</b>：未指定。
    * <b>优化算法</b>：设置为<code>0.01</code>，降低模型对变点的敏感度，减少过拟合。



* <code>seasonality_prior_scale</code>
    * <b>原算法</b>：未指定。
    * <b>优化算法</b>：设置为<code>10.0</code>，增强季节性特征的表达能力。







#### <b>新增参数</b>

* <code>seasonality_mode</code>
    * <b>优化算法</b>：引入参数，默认值为<code>additive</code>。



* <code>yearly_seasonality</code>
    * <b>优化算法</b>：支持按年建模，默认值为<code>True</code>。



* <code>weekly_seasonality</code>
    * <b>优化算法</b>：支持按周建模，默认值为<code>True</code>。



* <b>平滑因子 (</b><code>alpha</code><b>)</b>
    * <b>优化算法</b>：新增平滑因子，对预测结果进行指数平滑处理，默认值为<code>0.2</code>，减少预测波动。







#### <b>异常值处理</b>

* <b>原始版本</b>
    * 无。



* <b>优化算法</b>
    * 使用<code>Isolation Forest</code>模型检测异常值，标记并删除异常值。







#### <b>预测模型准确性评估</b>

<p style="text-align:start"><b>SMAPE：对称平均绝对百分比误差</b></p>

![](https://clouddocs.huawei.com/api/file/doc/preview/df53436e-b8b0-4ccd-bbf2-33618739388e?documentId=369acbe5-df2d-4b8f-afc1-82a59ec85d64 "")

<p style="text-align:unset"> </p>

<p style="text-align:unset"><b>SMAPE 的优点</b></p>

* 对称性：相比 MAPE，不会因预测值较小而导致极端百分比误差。
* 比例误差评估：更适合评估预测值与实际值之间的比例关系误差。
* 适应性：适用于不同规模的数据，具有较好的通用性。




<p style="text-align:unset"> </p>

<p style="text-align:unset"><b>SMAPE的范围：</b></p>

* SMAPE 的值范围是 0% 到 200%。
* 0%：表示预测值与实际值完全一致，没有误差。
* 数值越低表示误差越小。




<p style="text-align:unset"></p>

<p style="text-align:start"></p>

![](https://clouddocs.huawei.com/api/file/doc/preview/7eba630a-9465-443a-8b8b-49e03dead2fb?documentId=369acbe5-df2d-4b8f-afc1-82a59ec85d64 "")

![](https://clouddocs.huawei.com/api/file/doc/preview/7cfb2463-1fcb-45f7-9bf9-f43104bdfddb?documentId=369acbe5-df2d-4b8f-afc1-82a59ec85d64 "")

![](https://clouddocs.huawei.com/api/file/doc/preview/d4ff65c4-a718-4d46-a92c-581552d78161?documentId=369acbe5-df2d-4b8f-afc1-82a59ec85d64 "")

<p style="text-align:unset"></p>

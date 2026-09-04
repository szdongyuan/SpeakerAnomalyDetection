# VE3668N 选择后预录初始化设计

## 目标

每次应用进程启动期间最多执行一个 Vkinging VE3668N 不落盘预录周期，以主动创建并验证 SDK Task、IIO multi-buffer 和数据读取链路。一个周期包含首次尝试以及首次发生可恢复错误后的唯一一次自动重试。

唯一预录机会可以由两个入口中的首个有效 Vkinging 选择消费：

- 程序启动时恢复的已保存 Vkinging 选择；
- 运行中首次通过硬件设置确认的有效 Vkinging 选择。

预录机会一经开始、失败或因硬件忙被丢弃，本次进程生命周期内都视为已消费，后续设备切换或再次确认不得预录。若唯一触发因硬件忙被丢弃，则不排队、不补做，并允许正式录音和校准沿用现有懒初始化路径。若预录实际启动，第一次发生可恢复的原生初始化或采集错误时，程序完整回收失败 generation，等待 0.75 秒后自动重试一次；第二次仍失败时只阻止同一失败采集签名的正式录音和校准，并显示可诊断错误。程序本身继续运行，普通声卡及不同 VE 采集签名不继承该失败门禁，但也不会获得新的预录机会。

## 实现前提与基线

Git 清理锚点是工作流启动时记录的提交 `a9d2e8d293fae513113990a2b4e5cc7b2ca4dc99`。该提交只用于最终内容保留式历史清理，不表示其中已经包含当前工作区的 VE3668N 集成。

本设计依赖当前工作流临时检查点中的既有实现：

- `base/ve3668n_resource.py` 的实例级常驻 VE 原生资源控制器；
- `base/recording_worker.py`、`base/recording_service.py` 与 `ui/recording_service_bridge.py` 的进程隔离、generation 淘汰和 GUI 异步交付；
- `main_window.py` 的启动恢复、异步 VE 发现、硬件切换和准入控制；
- `ui/hardware_window.py` 的显式硬件选择保存；
- 当前 VE 主录音、校准、设备发现及相应测试。

如果计划或实现环境缺少这些前置模块，必须停止并报告基线错误，不得另建一套并行录音资源所有权。

## 根因与现有行为

Windows 冷启动会重置 USB 主控、libusbK、设备固件和 IIO 采集状态。VkDaqAssistant 能枚举设备并显示 `Available`，但这不证明 `VkDaqStartTask` 创建 multi-buffer 的链路已经成功。

现场日志显示 Windows 启动后即使已经等待十余分钟，第一次正式录音仍可能在 `VkDaqStartTask` 失败：

```text
VkDaqStartTask (code=-12001): iio_device_create_multi_buffer: invalid argument
```

随后重新创建任务或再次录音能够成功。这证明固定延长应用启动等待不能可靠解决问题；必须真正执行一次 Task 启动和数据读取，并在失败时完成资源回收后重建。

当前常驻资源控制器仍采用首次正式请求懒初始化。初始化失败会把控制器和 worker generation 标为不可复用，随后待绑定请求还可能收到 `VE resource owner exited before binding`。后者是同一次原生初始化失败的次生错误，不应替代首个 `VkDaqStartTask` 诊断。

## 范围

### 包含

- 每次应用进程启动一个实例级预录机会；
- 启动恢复或运行中硬件设置的首个有效 Vkinging 选择消费该机会；
- 机会被硬件忙丢弃后不再预录但允许正式录音；
- 使用当前已确认的设备、选中有序物理通道和采样率；
- 0.5 秒实际数据读取，数据只用于确认链路，不保存 WAV、不做校准、不做分析、不计数；
- 第一次可恢复失败后的 generation 级清理、0.5～1 秒退避和一次自动重试；
- 预录 available、pending、succeeded、skipped_busy、failed 状态及签名级失败门禁；
- 成功、失败、重试及首因诊断日志；
- 假 SDK 自动化验证和 Windows 真机验收。

### 不包含

- 修改 VkDaqAssistant 配置或依赖它执行录音任务；
- 普通声卡预录；
- 把预录数据写入用户录音目录、数据库、最近录音、报告或分析流水线；
- 自动切换到普通声卡；
- 无限重试、后台持续重连或新增守护进程；
- 通过固定等待替代实际预录；
- 与预热无关的录音、分析或硬件界面重构。

## 方案选择

### 采用：通过现有录音服务和 worker 执行专用 VE 预录命令

GUI 只发起异步预录意图。`RecordingService` 负责准入、generation、重试和终态；现有录音 worker 中的 VE 常驻控制器负责唯一原生 Task 所有权。预录成功后控制器保持健康空闲，正式录音继续复用同一签名和已启动 Task。

该方案不会在 GUI 线程调用 SDK，也不会创建第二个硬件所有者，能够复用现有进程死亡确认和资源不确定性门禁。

### 不采用：GUI 或主进程直接调用 VkDaq SDK

该方案会与录音 worker 争抢设备，使 UI 阻塞，并绕过现有 generation 淘汰、超时和关闭流程。

### 不采用：把第一次正式录音当作预热并在失败后重试

该方案让用户的正式测试继续承担首次失败，可能创建失败文件或推进业务状态，不满足“正式录制前先预录”的要求。

## 触发与去重

应用启动器在创建 `QApplication` 时创建一个 `VePrewarmLifetime` 实例，并把同一个对象注入主窗口及需要观察状态的 GUI 协调层。启动器持有该对象直到进程事件循环退出；它位于主窗口和录音服务之上，因此重建任一对象都必须复用同一个 lifetime。直接运行 `main_window.py` 的开发入口也必须在该入口创建一次并注入，不能由 `MainWindow` 构造函数偷偷创建新的默认机会。

`VePrewarmLifetime` 维护进程级机会状态 `available -> pending -> consumed`。它只在新应用进程的 bootstrap 中初始化为 `available`，不写入配置文件，也不因登录、打开/关闭硬件窗口、设备切换、选择 token 更新、MainWindow 重建、RecordingService/bridge 重建或 worker generation 重建而恢复。第一次有效 Vkinging 触发在 GUI 线程事务中通过该对象原子把机会从 `available` 改为 `pending`；成功、连续失败、确定性失败或 busy 丢弃都最终进入 `consumed`。不得使用可变模块全局变量、QApplication 动态属性或进程外持久化模拟此生命周期。

### 启动恢复

构造 `MainWindow` 时仍只恢复已保存的硬件快照，不进行同步原生 I/O。异步 VE 发现返回且满足以下条件后发起预录：

- 当前选择仍是 Vkinging；
- `resolve_ve_input` 确认设备 `available=true`；
- 选择包含非空、合法的有序物理通道；
- 采样率来自当前已验证输入配置；
- 应用未关闭，录音/校准/释放流程未占用硬件。

发现结果不可用时不尝试预录，维持现有不可用诊断和录音阻止行为。迟到的旧发现 generation 不得为已经变更的选择发起预录。

若有效发现结果到达时主录音、校准、VE 释放、worker 淘汰或其他硬件操作正在占用唯一硬件槽，主窗口直接消费并丢弃本次启动的唯一预录机会，不建立延迟意图、不排队，也不在硬件空闲后自动补做。当前预录结果进入 `skipped_busy`；正式录音和校准不因跳过预录而被阻止，仍受现有硬件忙和所有权安全门禁约束。该丢弃只写入状态栏和日志，不弹出模态错误框。

### 硬件设置确认

硬件对话框取消时不消费预录机会。用户点击“确定”并成功保存 Vkinging 选择后，只有实例级机会仍为 `available` 才进入以下预录事务；机会已经 `pending/consumed` 时仅应用硬件设置，后续正式录音沿用现有释放和懒初始化，不预录：

- 原签名不兼容且释放请求成功获得服务准入时，该释放属于同一次选择事务；只有释放成功回调到达后才创建预录触发。这不是对“硬件忙时预录触发”的延迟排队；释放完成前尚不存在预录触发；
- 没有旧 VE 资源或签名兼容时直接进入预录准入；
- 释放准入直接返回 busy 时，不延迟释放也不创建预录触发，唯一机会进入 `consumed/skipped_busy`；当前操作结束且服务确认没有所有权不确定状态后按原有规则开放录音和硬件设置，不得自动释放或预录；
- 释放已经获得准入但随后失败、服务 closing 或所有权不确定时不创建预录，按后文规定的 release/ownership failed 状态和通知处理。

每次成功确认 Vkinging 硬件设置仍产生新的“选择应用 generation”，但只有本次进程的实例级机会为 `available` 时才可能执行预录。机会已消费时，即使切换 machine ID、改变通道/采样率或再次确认相同签名，也不再预录。设备选择、通道勾选、采样率下拉框和发现刷新过程中产生的中间 Qt 信号不得消费机会或触发预录。

### 去重和过期结果

主窗口使用进程实例级机会和选择 token 关联唯一预录周期，不使用模块全局状态。同一进程只允许一个在途预录。回调必须同时匹配消费机会时的 token 和 VE 采集签名；旧 token、旧签名、关闭中的回调只记录或忽略，不能改变新选择的准入或弹出过期错误，也不能返还已经消费的机会。

## 专用预录协议

预录是独立的内部硬件操作，不伪装成主录音或校准，不复用需要 WAV 路径、校准元数据、结果 reader、文件租约或 UI 录音回调的 `RecordingRequest`。

父服务、worker 协议和控制器之间使用一个不可变的内部预录请求，至少冻结：

- 唯一 warmup ID；
- VE 设备快照及 machine ID；
- 有序物理通道；
- 采样率；
- 每通道目标帧数 `max(1, ceil(sample_rate * 0.5))`；
- 当前自动尝试序号。

目标量必须定义为“每通道帧数”，不得使用跨通道标量样本总数。一次原生读取返回值表示每通道帧数；N 通道交织缓冲区的有效标量数必须等于 `returned_frames * N`，容量必须等于 `requested_frames * N`。每次读取先验证返回帧数范围和交织缓冲区容量/类型，再按每通道累计帧数推进目标；部分读取可以累计，不能把通道数乘入进度，也不能把不足目标的部分读取误判为成功。

请求必须通过现有 VE 设备、通道、采样率和采集签名验证。预录成功的定义是：

1. SDK/设备解析成功；
2. Task、通道和采样时钟创建成功；
3. `VkDaqStartTask` 成功；
4. 验证实际采样率与请求一致；
5. 读取并验证累计达到 0.5 秒对应的每通道目标帧；
6. 预录适配器确认解绑，控制器回到健康 `IDLE`；
7. 不产生 WAV、预览、进度、录音结果或业务计数。

控制器进入 `IDLE` 后继续沿用现有有界空闲读取并丢弃数据，以保持 multi-buffer 和 Task 热状态。正式录音只在相同采集签名下复用。签名变化仍执行现有显式释放，但由于本次进程的预录机会已经消费，新签名在正式录音时按既有路径懒初始化，不重新预录。

内部协议增加稳定事件名时，应集中定义在现有协议边界，不能在多个模块复制字符串。

## 服务状态与并发

`RecordingService` 增加实例级预录操作状态，不改变现有公开录音方法的参数和回调。一次预录与下列操作互斥：

- 主录音硬件采集；
- VE 校准采集；
- VE 显式释放；
- worker generation 淘汰或关闭；
- 另一次预录。

预录占用现有唯一硬件采集槽，但不占用录音结果流水线容量，不创建文件租约或结果会话。服务的硬件忙状态必须在接受预录意图时原子建立，不能先返回 GUI 再异步占用，以免正式录音或硬件切换抢先开始。

预录期间：

- `can_start_recording=false`；
- `RecordingServiceBridge.hardware_busy=true`；
- 硬件设置、校准和正式录音入口均不可用；
- 关闭流程可以取消预录并取得有界资源释放确认；
- 不允许阻塞 Qt 线程等待 SDK、进程或退避时间。

预录终态必须异步回到 bridge 的 GUI 线程，并且每个 warmup ID 只交付一次最终成功或失败。

## 自动重试与资源恢复

### 可重试错误

以下外部边界的原生初始化或采集失败允许自动重试一次：

- SDK 加载/打开；
- 设备解析或设备属性读取；
- Task、IEPE 通道或采样时钟创建；
- `VkDaqStartTask`，包括现场 `code=-12001`；
- 实际采样率验证；
- 目标样本读取或原生缓冲区验证。

调用前的确定性参数错误、设备快照不可用、空通道、非法采样率、选择 token 已过期、服务 busy/closing 等不通过重复原生调用修复，因此不自动重试；它们直接返回对应准入或验证诊断。

### 第一次失败

第一次可重试失败不得在同一不确定控制器上重新 `StartTask`。服务必须：

1. 保留第一个原生异常的 stage、code 和 detail 为首因；
2. 使当前预录尝试终止；
3. 淘汰失败 worker generation；
4. 只有操作系统确认进程死亡并关闭父侧 process/IPC 句柄后，才认为该 generation 的原生所有权已经释放；
5. 使用非阻塞 deadline 退避 0.5～1 秒；
6. 生成新的 worker generation，用同一冻结签名执行第二次也是最后一次自动尝试。

若旧进程死亡无法确认，禁止重试和正式录音，返回资源所有权不确定诊断。

### 第二次失败

第二次失败后不再自动循环。服务完成有界淘汰并返回失败终态；GUI 把消费机会时的 VE 采集签名记录为 `failed_signature`，保留首因和第二次诊断。本次进程不再产生新的预录周期。同一失败签名的正式录音和校准保持禁用；原生所有权已确认释放时重新开放硬件设置，所有权仍不确定时继续安全锁定。普通声卡或不同 VE 采集签名不继承 `failed_signature` 门禁，可以按既有懒初始化路径使用，但不预录。

预录机会已经原子占用后出现的确定性参数或快照验证失败不进入原生重试，直接消费机会并把该采集签名记为 `failed_signature`。启动恢复的设备尚不可用时不消费机会，沿用状态栏和工具提示展示不可用原因；同一进程稍后收到首个有效发现结果时仍可消费唯一机会。用户在硬件对话框确认了无效 VE 选择时，继续由硬件对话框现有校验阻止保存，也不消费机会。

服务 busy 是唯一预录机会的终态：记录 `skipped_busy`，不执行原生重试、不保留待办意图，也不设置 `failed_signature`；硬件空闲且所有权安全后允许正式录音。服务 closing 时消费并静默丢弃机会，应用关闭期间不弹窗。预录之前的 VE 释放已经获得准入但随后失败时设置对应 `failed_signature`：若旧 worker 已确认死亡、原生所有权已释放，则弹一次释放首因错误并重新开放硬件设置；若 worker 死亡无法确认或所有权仍不确定，则弹一次错误并保持硬件设置禁用，提示关闭并重启应用。只有服务以后确认旧进程死亡并清除所有权不确定状态，才可重新开放硬件设置；不得因为 GUI 已进入失败终态就假定设备可安全重选。

`VE resource owner exited before binding` 只能作为附加诊断。若同一次尝试已经记录更早的 `VkDaqStartTask` 或其他原生失败，GUI 和日志必须优先展示首因。

## GUI 行为

进程级 `VePrewarmLifetime` 维护预录机会状态及可选的 `failed_signature`，主窗口只展示并执行准入：

```text
available       本次进程尚未消费唯一预录机会
pending         正在预录、重试、释放或等待 generation 淘汰
succeeded       唯一预录周期成功，机会已消费
skipped_busy    唯一触发因硬件忙被丢弃，机会已消费但不阻止后续正式录音
failed          实际预录/释放/验证失败，机会已消费，并记录对应 failed_signature
```

`failed` 必须同时保存失败类别、失败采集签名、精简首因、完整诊断以及 `ownership_safe` 布尔值。`skipped_busy` 不设置 `failed_signature`。

启动恢复 VE 时，状态栏在发现成功后显示“VE 设备正在初始化…”。硬件切换确认 VE 后，对话框可以正常关闭，主窗口显示相同状态；不得用模态等待阻塞硬件窗口关闭。

状态为 `pending` 时，正式录音、校准和硬件选择暂时禁用。状态为 `skipped_busy` 时不弹窗；底层硬件操作结束且所有权安全后，正式录音、校准和硬件设置按原有准入恢复，但不再预录。状态为 `failed` 时：

- 只有当前 VE 采集签名与 `failed_signature` 相同时，正式录音和校准被准入检查阻止；普通声卡或不同 VE 签名不受该失败门禁影响；
- `ownership_safe=true` 时硬件设置重新可用；`ownership_safe=false` 时硬件设置继续禁用并提示重启应用；
- 参数/快照验证失败、预录尝试失败或所有权安全的释放失败只弹出一次“Vkinging 设备初始化失败”错误框，显示精简首因并提示进入硬件设置重新选择；
- 所有权不确定的释放/淘汰失败只弹出一次错误框，显示精简首因并提示关闭和重启应用，不提示当前即可重选；
- 当前失败实际产生的完整诊断写入日志；两次尝试都发生时保留两次诊断，未进入重试的失败不得伪造第二次诊断；
- 不关闭程序、不清除用户已经保存的选择、不自动降级普通声卡。

状态为 `succeeded` 后恢复当前权限允许的正式录音、校准和硬件设置，预录成功不弹窗。机会已经 `succeeded` 或 `skipped_busy` 后切换到其他 Vkinging 签名时不再预录，正式录音使用既有懒初始化。现有所有正式录音入口（按钮、串口/硬件触发和其他已有自动入口）都必须经过同一 `pending`/`failed_signature` 准入，不能只禁用一个按钮。

## 超时

预录必须复用现有可注入的服务和控制器 deadline，并按以下确定规则执行；测试可注入更短的正有限值，但生产默认值不得另行发明：

- worker ready：`RecordingService._ready_timeout`，生产默认 10 秒；超时记为 `ready_timeout`，属于可重试的 worker/初始化失败；
- 预录命令发出到原生开始确认：`RecordingService._start_timeout`，生产默认 10 秒；控制器内部绑定确认同时受 `VeResourceController.bind_timeout` 的生产默认 3 秒限制，先到期者决定 `start_timeout` 或 `bind` 首因，二者都属于可重试失败；
- 有效数据采集：从原生开始时间构造现有 `VeCaptureDeadline`。0.5 秒目标的总 deadline 固定为开始时间加 `0.5 + max(5.0, 0.5 * 0.1) = 5.5` 秒，且连续 5 秒没有新增有效帧即 `capture_timeout`；属于可重试采集失败；
- 适配器解绑：现有 `VeResourceController.detach_timeout`，生产默认 0.5 秒；超时记为 `detach` 并进入 generation 淘汰，属于可重试失败，但只有确认旧进程死亡后才可重试；
- 预录前的不兼容 VE 显式释放：`RecordingService._release_timeout`，生产默认等于 5 秒 shutdown timeout；超时记为 `release_ve`，淘汰旧 generation，并按所有权是否确认决定能否重试；
- worker 淘汰：立即请求 terminate；到 `RecordingService._terminate_timeout` 的生产默认 2 秒仍未退出则执行 kill。只有 `_dead` 路径观察到操作系统确认进程死亡并关闭 process/IPC 句柄，才能重试或解除所有权门禁。到 2 秒 kill 点仍未确认死亡时，本次预录向 GUI 终结为所有权不确定失败，不进行第二次尝试；服务继续保持硬件和录音门禁，稍后死亡确认只能解除安全门禁，不能把旧 token 自动改成成功；
- 两次尝试间退避：实例级可注入 delay，生产值固定 0.75 秒，通过 supervisor deadline 实现，不调用阻塞 `sleep`；
- 应用关闭：沿用 `_shutdown_timeout=5` 秒，随后进入上述 terminate/2 秒 kill 流程。关闭终态不触发预录重试或用户错误弹窗。

0.5 秒是成功所需的有效采集数据时长，不是固定等待。所有 timeout 都保留当前尝试首因；清理、kill、bind 拒绝等次生错误只追加诊断。正常用户可见初始化通常应在 1～3 秒完成；首次失败并完成 generation 重建时可以更长，但由上述 deadline 严格限制。

## 异常边界

- SDK、设备和控制器错误在 worker 内保留原始 operation/stage、native code 和 detail；
- worker 协议把预录终态归一化为明确 success/failure，不把主录音结果事件用于预录；
- `RecordingService` 负责 generation、重试次数、退避、资源所有权和 exactly-once 终态；
- bridge 只负责把冻结终态排入 Qt GUI 线程；
- `MainWindow` 只负责当前选择 token、状态展示和业务准入。

禁止没有恢复价值的重复异常包装、宽泛吞噬、静默继续正式录音、静默切换声卡或在资源所有权不确定时创建新 worker。GUI 回调异常只能记录，不能反向杀死健康录音服务；服务仍必须清除相应预录 pending 状态。

## 自动化验证

使用假 SDK、可控 worker/bridge 和 Qt 测试验证：

- 每个应用进程只有一个由启动器持有的 `VePrewarmLifetime` 和一个预录机会，新进程才重置；登录、硬件窗口、MainWindow、RecordingService/bridge 和 worker 重建都不会重置；
- 用同一个 lifetime 依次重建 MainWindow 和 RecordingService/bridge，确认机会状态、outcome 和 `failed_signature` 保持不变；构造 MainWindow 未注入 lifetime 时明确失败而不是隐式创建第二个机会；
- 启动恢复普通声卡不消费机会；
- 启动恢复 VE 时，发现不可用不消费机会，首个有效发现结果消费并最多执行一个预录周期；
- 有效发现结果在硬件忙时消费并丢弃唯一机会，不建立延迟意图、不在空闲后自动预录，记录 `skipped_busy` 且后续正式录音仍允许；
- 迟到发现结果或过期 token 不触发或不改变当前选择；
- 硬件对话框取消不预录；
- 每次确认 VE 选择可以创建新 token，但只有实例机会为 `available` 的首个有效选择能消费并触发预录；中间设备/通道/采样率信号不消费机会；
- 唯一机会消费后，从普通声卡切回 VE、切换 VE machine ID、改变通道顺序/集合、改变采样率和再次确认相同签名都不再预录；
- 不兼容旧 VE 先完成异步释放，随后才预录；释放失败时不并行启动预录；
- 预录冻结并使用当前 machine ID、有序通道和采样率；
- 预录累计读取精确目标至少 0.5 秒的数据，确认解绑后成功；
- 预录不创建 WAV、临时录音文件、reader、文件租约、预览、分析、最近录音、数据库或计数事件；
- 成功后控制器保持健康 `IDLE`，相同签名正式录音复用同一 SDK/Task，不重复 `CreateTask`/`StartTask`；
- pending 时所有正式录音入口、校准和硬件选择被阻止，Qt 线程不阻塞；
- succeeded 后恢复准入；skipped_busy 在底层忙状态结束后允许正式录音且不再预录；failed 只阻止匹配 `failed_signature` 的录音/校准；
- 启动恢复设备不可用和 skipped_busy 都不弹模态框；其余 failed 类别各自只执行一次规定弹窗，所有权不确定提示重启而不是重选；
- busy 丢弃后硬件槽变为空闲也绝不自动释放或预录；服务确认所有权安全后恢复原有准入，重新确认设备也不会返还机会；
- 首次 `VkDaqStartTask -12001` 失败保留为首因，失败 generation 确认死亡后退避并只重试一次；
- 首次读取等其他可恢复原生错误走相同一次重试；
- 参数验证、设备不可用、busy/closing 不执行无意义的自动重试；busy 触发消费机会并进入 skipped_busy，closing 静默结束；
- 参数/快照失败、busy 触发丢弃、释放失败和所有权不确定分别进入规定状态、通知策略、签名门禁和硬件重选门禁；
- 第二次失败只交付一次失败弹窗，保留两次诊断，不开始正式录音；
- 第二次失败只阻止与 `failed_signature` 完全相同的 VE 正式录音/校准；普通声卡及不同 VE 签名允许使用既有懒初始化，且不获得第二次预录；
- `owner exited before binding` 不覆盖更早首因；
- 旧 generation 死亡无法确认时不重试、不开放录音；
- 关闭期间取消在途预录并执行有界释放，无迟到弹窗或状态复活；
- 多通道部分读取按每通道帧累计，严格验证 `returned_frames * channel_count` 的交织缓冲区，不把标量样本数当作帧数；
- ready、start/bind、5.5 秒采集、5 秒无进度、0.5 秒 detach、5 秒 release、2 秒 terminate/kill、0.75 秒退避和 5 秒 shutdown 均使用可注入时钟进行确定性边界测试；
- 普通声卡主录音、现有 VE 主录音、VE 校准、持久资源、硬件切换和关闭测试继续通过。

## 真机验收

在发生过现场问题的 Windows/VE3668N USB 环境执行：

1. 完整重启 Windows，不手工重启 VkDaqAssistant；
2. 保持已保存输入为 Vkinging，启动应用；
3. 观察启动发现后唯一自动预录周期达到 `succeeded`；
4. 立即执行正式录音，确认不出现 `-12001` 和次生 bind 错误；
5. 切换普通声卡后再切回 Vkinging，确认本次进程不再次预录，正式录音走既有懒初始化；
6. 改变 VE 通道和采样率，分别确认旧签名释放，但不再次预录，正式录音走既有懒初始化；
7. 注入或模拟第一次 `-12001`，确认只自动重试一次且首因日志保留；
8. 注入连续两次失败，确认只有匹配 `failed_signature` 的正式录音/校准被阻止；普通声卡及不同 VE 签名仍可使用但不再次预录，硬件设置在所有权安全后可重新打开，且没有失败 WAV。

至少连续完成 10 次“启动新应用进程、唯一预录周期结束后立即正式录音”的循环。另覆盖唯一机会被 busy 丢弃后允许正式录音，以及同一进程多次切换 Vkinging 仍没有第二次预录。记录进程实例、机会状态、选择 token、采集签名、warmup ID、worker generation、每次尝试耗时、首因、生命周期计数及是否产生用户文件。

## Compatibility / Migration

- **Backward compatibility: partial**
- **Protected surfaces:** `[existing RecordingService recording method signatures and callbacks, normal sound-card behavior, VE main/calibration WAV data and metadata, hardware selection file format, user recordings/database/reports, persistent VE signature rules, bounded shutdown and generation ownership]`
- **Allowed breakage:** `[Vkinging selection now has an asynchronous pending state before recording is allowed, hardware selection confirmation may be followed by visible initialization status, VE formal recording/calibration is intentionally blocked only when its acquisition signature matches failed_signature, additive process-lifetime prewarm owner and private prewarm service/bridge/worker protocol and state]`
- **Migration strategy: compatibility layer**

不迁移配置、数据库、WAV 或用户数据。已有调用方的录音 API 保持不变；生产 GUI 在既有准入之外增加 `pending` 期间阻止全部硬件操作、当前签名匹配 `failed_signature` 时阻止 VE 录音/校准的条件。`skipped_busy` 在底层硬件安全后允许正式录音；普通声卡不经过新增门禁。

## 实现约束

- 使用由应用启动器创建和持有的进程生命周期 `VePrewarmLifetime`，以及实例级选择 token 和重试状态；MainWindow/服务重建必须复用它，不增加可变模块全局硬件状态。
- 所有 VkDaq SDK 调用继续只由现有 worker 内的唯一 VE 所有者线程执行。
- 不引入与预热无关的重构。
- 预录数据不得进入任何用户或业务持久化路径。
- 异步操作、进程等待、退避和关闭必须有界，不阻塞 Qt 线程。
- 新增异常必须具有语义、诊断或恢复价值；失败在上述责任边界处理或传播。
- 禁止不合理的宽泛捕获、静默回退、重复防御包装和用睡眠猜测原生资源已经释放。
- 不引入单次使用的模块级常量。只有稳定的内部协议字段、序列化状态或需要跨模块及测试引用的边界值才可按项目惯例集中定义；其他值使用局部变量或实例字段。
- 保留首个原生失败的 operation、code 和 detail；次生清理或 bind 错误只追加诊断。
- 所有成功、失败、重试、释放和过期回调都必须按 warmup ID、选择 token、签名和 worker generation 精确归属。

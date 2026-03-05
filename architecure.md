# OptiSim architecture 

## 1. Class diagram


```mermaid
classDiagram
    class SimulationApp {
        -SimulationController controller
        -MainWindow main_window
        -list open_windows
        +start()
        +handle_simulation_finished(payload)
        +handle_error(err_msg)
        +open_strategy_window(track_name)
    }

    class MainWindow {
        -SimulationController controller
        -LoadingOverlay loading_overlay
        +openCompareRequested: pyqtSignal
        +openStrategyRequested: pyqtSignal
        +on_run_single_clicked()
        +on_lap_map_clicked()
        -_get_car_params() dict
    }

    %% Data Transfer Objects (DTOs)
    class BaseRequest {
        +dict car_params
        +float ds
    }
    class SingleLapRequest
    class EnduranceRequest
    class LapMapTestRequest
    class StrategyOptRequest
    class DeploymentOptRequest
    
    BaseRequest <|-- SingleLapRequest
    BaseRequest <|-- EnduranceRequest
    BaseRequest <|-- LapMapTestRequest
    BaseRequest <|-- StrategyOptRequest
    StrategyOptRequest <|-- DeploymentOptRequest

    class SimulationController {
        -SimulationWorker active_worker
        +simulationStarted: pyqtSignal
        +simulationProgress: pyqtSignal
        +simulationFinished: pyqtSignal
        +simulationError: pyqtSignal
        +handle_request(req: BaseRequest)
        -_create_solver(...)
    }

    class SimulationWorker {
        -SimulationTask task
        +progress: pyqtSignal
        +finished: pyqtSignal
        +error: pyqtSignal
        +run()
    }

    %% Tasks
    class SimulationTask {
        <<Abstract>>
        +execute(progress_callback) SolverPayload
    }
    class SingleLapTask
    class LaptimeMapTask
    class StrategyOptTask
    class DeploymentOptTask
    
    SimulationTask <|-- SingleLapTask
    SimulationTask <|-- LaptimeMapTask
    SimulationTask <|-- StrategyOptTask
    SimulationTask <|-- DeploymentOptTask

    class SolverPayload {
        +str kind
        +object data
        +dict meta
    }

    %% Relationships
    SimulationApp *-- SimulationController : Holds instance
    SimulationApp *-- MainWindow : Holds instance
    
    MainWindow ..> BaseRequest : Creates DTO
    MainWindow --> SimulationController : Calls handle_request(req)
    
    SimulationController ..> SimulationTask : Creates based on Request
    SimulationController *-- SimulationWorker : Manages thread
    
    SimulationWorker o-- SimulationTask : Executes
    SimulationTask ..> SolverPayload : Returns
```

## 2. State diagram 


```mermaid

stateDiagram-v2
    [*] --> Idle: Application waits for user

    state "UI Layer (MainWindow / StrategyWindow)" as UI {
        Idle --> ReadInput: Button click (e.g., Run Single Lap)
        ReadInput --> CreateRequest: Create DTO (e.g., SingleLapRequest)
        CreateRequest --> SendRequest: controller.handle_request(req)
    }

    state "Controller Layer (Assembly)" as Controller {
        SendRequest --> AssembleCar: Initialize Car() from req.car_params
        AssembleCar --> Routing: What type of Request is it?
        
        Routing --> CreateSingleTask: type(req) is SingleLapRequest
        Routing --> CreateLapMapTask: type(req) is LapMapTestRequest
        Routing --> CreateStrategyTask: type(req) is StrategyOptRequest
        Routing --> CreateDeploymentTask: type(req) is DeploymentOptRequest
        
        CreateSingleTask --> StartWorker
        CreateLapMapTask --> StartWorker
        CreateStrategyTask --> StartWorker
        CreateDeploymentTask --> StartWorker
        
        StartWorker --> active_worker.start()
    }

    state "Loading (MainWindow)" as Loading {
        AssembleCar --> ShowLoading: emit simulationStarted()
    }

    state "Worker Thread (SimulationWorker)" as Worker {
        active_worker.start() --> TaskExecute: task.execute(callback)
        TaskExecute --> TaskExecute: emit progress(current, total)
        TaskExecute --> CreatePayload: Computation finished
    }

    state "Results Processing (SimulationApp)" as Results {
        CreatePayload --> Finish: emit finished(SolverPayload)
        
        Finish --> CreateWindow: App catches signal and checks payload.kind
        CreateWindow --> ShowWindow: App stores and shows window (Results, Sensitivity, etc.)
    }

    ShowWindow --> Idle
    Finish --> HideLoading: UI catches signal and hides overlay
    HideLoading --> Idle
```
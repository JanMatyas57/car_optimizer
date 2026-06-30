# Laptime simulator & strategy optimizer

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd optiSim
   ```

2. **Set up a virtual environment (Optional but recommended)**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Main GUI

To launch the primary graphical user interface:

```bash
python main.py
```

### Running standalone sweep scripts

You can bypass the UI and call the simulation from a python script. See the `utils/` directory for examples.  

For example, to generate a mass vs. CD sensitivity grid:
```bash
python utils/run_mass_cd_grid.py
```

To generate a full battery endurance grid for the strategy optimizer:
```bash
python utils/run_battery_laptime_grid.py
```


## Architecture


### 1. Class diagram

```mermaid
classDiagram
    class SimulationApp {
        -SimulationController controller
        -MainWindow main_window
        -list open_windows
        +start()
        +handle_simulation_finished(payload)
        +handle_error(err_msg)
        +open_strategy_picker()
        +open_compare_window()
    }

    class MainWindow {
        -SimulationController controller
        -LoadingOverlay loading_overlay
        +openCompareRequested: pyqtSignal
        +openStrategyRequested: pyqtSignal
        +on_run_single_clicked()
        +on_sensitivity_clicked()
        -_get_car_params() dict
        +finish_track_generation(data)
    }

    %% Data Transfer Objects (DTOs)
    class BaseRequest {
        +dict car_params
        +float ds
    }
    class SingleLapRequest
    class EnduranceRequest
    class StrategyOptRequest
    class DeploymentOptRequest
    class TrackGenerationRequest
    class OptimizationRequest
    class SensitivityRequest
    
    BaseRequest <|-- SingleLapRequest
    BaseRequest <|-- EnduranceRequest
    BaseRequest <|-- StrategyOptRequest
    BaseRequest <|-- TrackGenerationRequest
    BaseRequest <|-- OptimizationRequest
    BaseRequest <|-- SensitivityRequest
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
    class StrategyOptTask
    class DeploymentOptTask
    class TrackGenerationTask
    class OptimizationTask
    class LaptimeSensitivityTask
    
    SimulationTask <|-- SingleLapTask
    SimulationTask <|-- StrategyOptTask
    SimulationTask <|-- DeploymentOptTask
    SimulationTask <|-- TrackGenerationTask
    SimulationTask <|-- OptimizationTask
    SimulationTask <|-- LaptimeSensitivityTask

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

### 2. State diagram 

```mermaid
stateDiagram-v2
    [*] --> Idle: Application waits for user

    state "UI Layer (MainWindow / Dialogs)" as UI {
        Idle --> ReadInput: Button click (e.g., Run Single Lap, Track Gen)
        ReadInput --> CreateRequest: Create DTO (e.g., SingleLapRequest, OptimizationRequest, SensitivityRequest)
        CreateRequest --> SendRequest: controller.handle_request(req)
    }

    state "Controller Layer (Assembly)" as Controller {
        SendRequest --> AssembleCar: Initialize Car() (if needed)
        AssembleCar --> Routing: Route by Request type
        Routing --> CreateSingleTask: SingleLapRequest
        Routing --> CreateStrategyTask: StrategyOptRequest
        Routing --> CreateDeploymentTask: DeploymentOptRequest
        Routing --> CreateTrackGenTask: TrackGenerationRequest
        Routing --> CreateOptimizationTask: OptimizationRequest
        Routing --> CreateSensitivityTask: SensitivityRequest
        
        CreateSingleTask --> StartWorker
        CreateStrategyTask --> StartWorker
        CreateDeploymentTask --> StartWorker
        CreateTrackGenTask --> StartWorker
        CreateOptimizationTask --> StartWorker
        CreateSensitivityTask --> StartWorker
        
        StartWorker --> active_worker.start()
    }

    state "Loading (MainWindow/Overlays)" as Loading {
        AssembleCar --> ShowLoading: emit simulationStarted()
    }

    state "Worker Thread (SimulationWorker)" as Worker {
        active_worker.start() --> TaskExecute: task.execute(callback)
        TaskExecute --> TaskExecute: emit progress(current, total)
        TaskExecute --> CreatePayload: Computation finished
    }

    state "Results Processing (SimulationApp)" as Results {
        CreatePayload --> Finish: emit finished(SolverPayload)
        
        Finish --> ProcessPayload: App checks payload.kind
        ProcessPayload --> ShowWindow: Kind is single_lap / endurance
        ProcessPayload --> UpdateMapTab: Kind is endurance_map / endurance_deployment
        ProcessPayload --> ShowOptDialog: Kind is optimization
        ProcessPayload --> ShowSensitivityDialog: Kind is sensitivity_analysis
        ProcessPayload --> MainWinFinishGen: Kind is track_generation
    }

    ShowWindow --> Idle
    UpdateMapTab --> Idle
    ShowOptDialog --> Idle
    ShowSensitivityDialog --> Idle
    MainWinFinishGen --> Idle
    
    Finish --> HideLoading: UI catches signal and hides overlay
    HideLoading --> Idle
```

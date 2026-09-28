param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('start', 'stop', 'status')]
    [string] $Action,

    [string] $PythonPath = 'python.exe',
    [string] $ReferenceRoot = '.runtime/xianyu-template',
    [string] $Database = 'logs/xianyu_stage3.sqlite3',
    [string] $LogFile = 'logs/xianyu_stage3.log',
    [string] $AcceptanceReport,
    [string] $Account,
    [switch] $EnableAccount,
    [int] $StopTimeoutSeconds = 120
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runnerPath = Join-Path $projectRoot 'scripts/run_xianyu_stage3.py'
$logsPath = Join-Path $projectRoot 'logs'
$pidPath = Join-Path $logsPath 'xianyu_stage3.pid'
$stopPath = Join-Path $logsPath 'xianyu_stage3.stop'
$stdoutPath = Join-Path $logsPath 'xianyu_stage3.stdout.log'
$stderrPath = Join-Path $logsPath 'xianyu_stage3.stderr.log'

New-Item -ItemType Directory -Path $logsPath -Force | Out-Null

function Get-RunnerProcess {
    if (-not (Test-Path -LiteralPath $pidPath)) {
        return $null
    }
    $rawPid = (Get-Content -LiteralPath $pidPath -Raw).Trim()
    $processId = 0
    if (-not [int]::TryParse($rawPid, [ref] $processId)) {
        Remove-Item -LiteralPath $pidPath -Force
        return $null
    }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
    if ($null -eq $process -or [string]::IsNullOrEmpty($process.CommandLine) -or -not $process.CommandLine.Contains($runnerPath)) {
        Remove-Item -LiteralPath $pidPath -Force
        return $null
    }
    return $process
}

function ConvertTo-QuotedArgument([string] $Value) {
    return '"' + $Value.Replace('"', '\"') + '"'
}

function Resolve-ProjectPath([string] $Value) {
    if ([IO.Path]::IsPathRooted($Value)) {
        return [IO.Path]::GetFullPath($Value)
    }
    return [IO.Path]::GetFullPath((Join-Path $projectRoot $Value))
}

if ($Action -eq 'status') {
    $process = Get-RunnerProcess
    if ($null -eq $process) {
        Write-Output 'Xianyu listener is stopped.'
        exit 1
    }
    Write-Output "Xianyu listener is running. PID=$($process.ProcessId)"
    Write-Output "Log: $([IO.Path]::GetFullPath((Join-Path $projectRoot $LogFile)))"
    exit 0
}

if ($Action -eq 'stop') {
    $process = Get-RunnerProcess
    if ($null -eq $process) {
        Write-Output 'Xianyu listener is already stopped.'
        exit 0
    }
    Set-Content -LiteralPath $stopPath -Value 'stop' -Encoding Ascii
    $deadline = [DateTime]::UtcNow.AddSeconds($StopTimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        Start-Sleep -Milliseconds 500
        $process = Get-RunnerProcess
        if ($null -eq $process) {
            Write-Output 'Xianyu listener stopped cleanly.'
            exit 0
        }
    }
    throw "Listener did not stop within $StopTimeoutSeconds seconds. It was left running; inspect $LogFile."
}

if ($null -ne (Get-RunnerProcess)) {
    throw 'Xianyu listener is already running. Use status or stop first.'
}
if ([string]::IsNullOrWhiteSpace($AcceptanceReport)) {
    throw 'start requires -AcceptanceReport for the current approved project commit.'
}

$python = (Get-Command $PythonPath -ErrorAction Stop).Source
$referencePath = Resolve-ProjectPath $ReferenceRoot
$databasePath = Resolve-ProjectPath $Database
$logFilePath = Resolve-ProjectPath $LogFile
$acceptancePath = Resolve-ProjectPath $AcceptanceReport
foreach ($path in @($runnerPath, $referencePath, $acceptancePath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path does not exist: $path"
    }
}
Remove-Item -LiteralPath $stopPath -Force -ErrorAction SilentlyContinue

$arguments = @(
    '-u', $runnerPath,
    '--project-root', $projectRoot,
    '--reference-root', $referencePath,
    '--db', $databasePath,
    '--log-file', $logFilePath,
    '--acceptance-report', $acceptancePath
)
if (-not [string]::IsNullOrWhiteSpace($Account)) {
    $arguments += @('--account', $Account)
}
if ($EnableAccount) {
    $arguments += '--enable'
}
$argumentLine = ($arguments | ForEach-Object { ConvertTo-QuotedArgument ([string] $_) }) -join ' '
$started = Start-Process -FilePath $python -ArgumentList $argumentLine `
    -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
Start-Sleep -Seconds 2
$started.Refresh()
if ($started.HasExited) {
    throw "Listener exited during startup (code $($started.ExitCode)). Inspect $stderrPath and $LogFile."
}
Set-Content -LiteralPath $pidPath -Value $started.Id -Encoding Ascii
Write-Output "Xianyu listener started. PID=$($started.Id)"
if ($EnableAccount) {
    Write-Output 'Account auto replies were explicitly enabled for this start.'
} else {
    Write-Output 'Account state was preserved; a paused account remains listen-only.'
}

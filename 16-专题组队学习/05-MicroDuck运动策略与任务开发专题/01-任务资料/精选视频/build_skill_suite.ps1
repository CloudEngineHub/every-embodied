$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Clips = Join-Path $Root 'clips'
$Out = Join-Path $Root 'microduck_skill_suite_v0.mp4'
$Concat = Join-Path $Clips 'concat.txt'
$Font = 'C\:/Windows/Fonts/msyh.ttc'

New-Item -ItemType Directory -Force $Clips | Out-Null

function Invoke-Ffmpeg([string[]]$FfmpegArgs) {
    & ffmpeg @FfmpegArgs
    if ($LASTEXITCODE -ne 0) {
        throw "ffmpeg failed with exit code $LASTEXITCODE"
    }
}

function New-TitleCard([string]$Path, [int]$Duration, [string]$Title, [string]$Subtitle) {
    $vf = "drawtext=fontfile='$Font':text='$Title':fontcolor=0xF4F8F8:fontsize=54:x=80:y=270"
    $vf += ",drawtext=fontfile='$Font':text='$Subtitle':fontcolor=0x8FC7C5:fontsize=26:x=82:y=355"
    Invoke-Ffmpeg @(
        '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'lavfi', '-i', "color=c=0x071318:s=1280x720:r=30:d=$Duration",
        '-vf', $vf, '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', $Path
    )
}

function New-Clip([string]$Path, [string]$Source, [double]$Start, [double]$Duration, [string]$Title, [string]$Status) {
    $vf = 'scale=1280:720:force_original_aspect_ratio=decrease'
    $vf += ',pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x071318'
    $vf += ',setsar=1'
    $vf += ',drawbox=x=0:y=0:w=1280:h=92:color=0x071318@0.86:t=fill'
    $vf += ",drawtext=fontfile='$Font':text='$Title':fontcolor=0xF4F8F8:fontsize=30:x=54:y=25"
    $vf += ",drawtext=fontfile='$Font':text='$Status':fontcolor=0xFFB45A:fontsize=22:x=930:y=30"
    Invoke-Ffmpeg @(
        '-hide_banner', '-loglevel', 'error', '-y', '-ss', $Start.ToString('0.###', [Globalization.CultureInfo]::InvariantCulture),
        '-i', $Source, '-t', $Duration.ToString('0.###', [Globalization.CultureInfo]::InvariantCulture),
        '-vf', $vf, '-an', '-r', '30', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', $Path
    )
}

$WalkSource = Join-Path $env:TEMP 'microduck-walking.mp4'
if (-not (Test-Path -LiteralPath $WalkSource)) {
    throw "Missing obstacle-walk source: $WalkSource"
}

$Repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $Root))
$Sources = @{
    Drag = Join-Path $Repo '07-机器人操作、运动控制\Locomotion\02-mjswan-MicroDuck浏览器物理扰动\assets\mjswan_microduck_manual_drag_demo.mp4'
    Ball = Join-Path $Repo '07-机器人操作、运动控制\Locomotion\05-MotrixLab-MicroDuck球平衡与FastSAC\assets\motrix_microduck_ball_balance_official.mp4'
    Swing = Join-Path $Repo '07-机器人操作、运动控制\Locomotion\04-MicroDuck摆动旋转强化学习复现\assets\microduck_swing_alpha050_local.mp4'
    Ladder = Join-Path $Repo '07-机器人操作、运动控制\Locomotion\06-MicroDuck梯面攀爬强化学习导读\assets\microduck_ladder_v1_local.mp4'
}
foreach ($source in $Sources.Values) {
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Missing showcase source: $source"
    }
}

Copy-Item -LiteralPath $WalkSource -Destination (Join-Path $Clips 'source_obstacle_walk.mp4') -Force

$Parts = @()
$title = Join-Path $Clips '00_title.mp4'
New-TitleCard $title 3 'MicroDuck 技能串联演示' '当前可复现结果 · 物理仿真与强化学习路线'
$Parts += $title

$clip = Join-Path $Clips '01_obstacle_walk.mp4'
New-Clip $clip $WalkSource 0 8.6 '01  绕障行走' '真实策略回放'
$Parts += $clip

$clip = Join-Path $Clips '02_physical_perturbation.mp4'
New-Clip $clip $Sources.Drag 18 12 '02  物理扰动' '引擎级接触演示'
$Parts += $clip

$clip = Join-Path $Clips '03_ball_balance.mp4'
New-Clip $clip $Sources.Ball 0 10 '03  球面平衡' 'FastSAC 策略回放'
$Parts += $clip

$clip = Join-Path $Clips '04_swing_contact.mp4'
New-Clip $clip $Sources.Swing 0 12 '04  摆动与接触' 'MuJoCo 动力学演示'
$Parts += $clip

$clip = Join-Path $Clips '05_ladder_contact.mp4'
New-Clip $clip $Sources.Ladder 0 10 '05  梯面接触' '阶段性结果 · 尚未登顶'
$Parts += $clip

$end = Join-Path $Clips '06_end_card.mp4'
$vf = "drawtext=fontfile='$Font':text='组合路线的当前状态':fontcolor=0xF4F8F8:fontsize=42:x=80:y=150"
$vf += ",drawtext=fontfile='$Font':text='已验证  绕障行走 · 物理扰动 · 球面平衡 · 梯面真实接触':fontcolor=0x8FC7C5:fontsize=25:x=82:y=270"
$vf += ",drawtext=fontfile='$Font':text='训练中  GroundPick · 连续换档 · 顶部落台':fontcolor=0xFFB45A:fontsize=25:x=82:y=330"
$vf += ",drawtext=fontfile='$Font':text='本片是阶段演示  不把 smoke 或部分回放写成最终成功':fontcolor=0xB8C4C5:fontsize=22:x=82:y=430"
Invoke-Ffmpeg @(
    '-hide_banner', '-loglevel', 'error', '-y',
    '-f', 'lavfi', '-i', 'color=c=0x071318:s=1280x720:r=30:d=5',
    '-vf', $vf, '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', $end
)
$Parts += $end

$concatLines = $Parts | ForEach-Object {
    $path = $_ -replace '\\', '/'
    "file '$path'"
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines($Concat, $concatLines, $utf8NoBom)
$concatArgs = @(
    '-hide_banner', '-loglevel', 'error', '-y',
    '-f', 'concat', '-safe', '0', '-i', $Concat,
    '-an', '-c', 'copy', '-movflags', '+faststart', $Out
)
Invoke-Ffmpeg $concatArgs

$probe = ffprobe -v error -show_entries format=duration:stream=width,height,codec_name -of default=nw=1 $Out
Write-Output "Created: $Out"
Write-Output ($probe -join "`n")


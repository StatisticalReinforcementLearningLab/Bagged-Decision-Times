library(dplyr)
library(rjson)
library(lubridate)

# dropbox = "/Users/daiqigao/Dropbox (Harvard University)/"
dropbox = "/Users/dqgao/Library/CloudStorage/Dropbox-HarvardUniversity/"
code.folder = paste0(dropbox, "Research/Projects/HS V5/RealData")
data.folder = paste0(dropbox, "*Shared/HeartStepsV2V3/Daiqi/Data/")
csv.folder = paste0(data.folder, "csv files/")
second.proc.folder = paste0(data.folder, "second_processing/")
third.proc.folder = paste0(data.folder, "third_processing/unwinsorized_data/")

## the folder "processed_data" is copied from "HeartStepsV2V3/Prasidh/second_processing/processed_data"
processed.data.folder = paste0(second.proc.folder, "processed_data/")

## the folder "final_data" is copied from "HeartStepsV2V3/Prasidh/second_processing/final_data"
load(paste0(second.proc.folder, "final_data/Idtable.Rdata"))

aims = c("aim-2", "aim-3")

aim2.userid = c(
  10006, 10008, 10015, 10027, 10032, 10041, 10044, 10047, 10055, 10062, 10075, 10086, 
  10094, 10101, 10105, 10110, 10118, 10124, 10137, 10142, 10152, 10156, 10157, 10161, 
  10163, 10172, 10178, 10186, 10187, 10194, 10195, 10199, 10214, 10215, 10217, 10218, 
  10234, 10237, 10238, 10259, 10261, 10262, 10269, 10271, 10283, 10293, 10296, 10304, 
  10307, 10310, 10313, 10327, 10336, 10339, 10342, 10352, 10355, 10360, 10365, 10374, 
  10388, 10389, 10395, 10399
)

aim3.userid = c(
  10033, 10037, 10040, 10042, 10043, 10054, 10063, 10069, 10092, 10132, 10138, 10154, 
  10162, 10175, 10182, 10198, 10200, 10210, 10211, 10221, 10232, 10267, 10270, 10280, 
  10287, 10332, 10343, 10356, 10366, 10376, 10394, 10416, 10425, 10426, 10439, 10451, 
  10455, 10461, 10470, 10495, 10514, 10523, 10562, 10575, 10584, 10589, 10622, 10625, 
  10628, 10643, 10645, 10659, 10665, 10674, 10680, 10681, 10698, 10707, 10710, 10718, 
  10723, 10731, 10737
)

userid.by.aim = list(aim2.userid, aim3.userid)
names(userid.by.aim) = aims


day.to.date = function(userid) {
  user.day = list()
  path = paste0(csv.folder, "kpwhri.", aim, ".", userid, ".walking-suggestion-service-requests.csv")
  dat.walk = read.csv(path, header = T)
  if (nrow(dat.walk) == 0) {
    next
  }
  for (j in 1:nrow(dat.walk)) {
    ## nightly-update rows contain the information about both the date and the day in study
    ## some users (10027, 10137, 10307) have multiple dates corresponding to the same day of study
    ## use later records to update the previous records
    if(as.character(dat.walk$url[j]) == "http://walking-suggestion:8080/nightly"){
      json.info = fromJSON(dat.walk$request_data[j])
      if (is.null(json.info$date) | is.null(json.info$studyDay)) {
        next
      }
      user.day[[json.info$studyDay]] = c(dat.walk$user__username[j], json.info$date, json.info$studyDay)
    }
  }
  user.day = data.frame(do.call(rbind, user.day))
  colnames(user.day) = c("userid", "date", "day")
  user.day$date = ymd(user.day$date)
  user.day$day = as.integer(user.day$day)
  user.day = unique(user.day)
  first.date = user.day[user.day$day == 1, "date"]
  for (d in -7:-1) { ## 1 week before study begins
    user.day = rbind(user.day, c(userid, first.date + d, d))
  }
  user.day = user.day %>% arrange(userid, day)
  return(user.day)
}


app.view = function(userid) {
  path = paste0(csv.folder, "kpwhri.", aim, ".", userid, ".client-page-views.csv")
  daily.metrics = read.csv(path, header = T)
  if (nrow(daily.metrics) == 0) {
    daily.metrics = data.frame(matrix(0, nrow = 0, ncol = 3))
  } else {
    daily.metrics$Date = ymd(daily.metrics$Date)
    daily.metrics = daily.metrics %>%
      group_by(Date) %>%
      summarise(app.view = n())
    daily.metrics = cbind(userid, daily.metrics)
  }
  colnames(daily.metrics) = c("userid", "date", "app.view")
  return(daily.metrics)
}


fitbit.worn = function(userid) {
  path = paste0(csv.folder, "kpwhri.", aim, ".", userid, ".daily-metrics.csv")
  daily.metrics = read.csv(path, header = T)
  daily.metrics = subset(daily.metrics, select = c(Date, Fitbit.Worn))
  daily.metrics = cbind(userid, daily.metrics)
  colnames(daily.metrics) = c("userid", "date", "fitbit.worn")
  daily.metrics$date = ymd(daily.metrics$date)
  return(daily.metrics)
}


fitbit.minutes.worn = function(userid) {
  ## step counts per minute
  path = paste0(csv.folder, "kpwhri.", aim, ".", userid, ".fitbit-data-per-minute.csv")
  steps.per.minute = read.csv(path, header = T)
  ## some users have rows with time like "17:24:S"
  steps.per.minute$time = gsub("S", "00", steps.per.minute$time)
  steps.per.minute$time = hms(steps.per.minute$time)
  ## dates
  if (grepl("/", steps.per.minute$date[1])) {
    ## for dates like "23/06/2019" (user 10006)
    steps.per.minute$date = dmy(steps.per.minute$date)
  } else {
    ## for dates like "2019-06-28" (user 10008)
    steps.per.minute$date = ymd(steps.per.minute$date)
  }
  
  ## from 6 a.m. to 11 p.m.
  day_time = (steps.per.minute$time >= hms("06:00:00")) & (steps.per.minute$time <= hms("23:00:00"))
  steps.per.minute = steps.per.minute[day_time,]
  steps.per.minute = steps.per.minute %>%
    group_by(date) %>%
    summarise(fitbit.minutes.worn = sum(!is.na(heart_rate)))
  steps.per.minute = subset(steps.per.minute, select = c(date, fitbit.minutes.worn))
  steps.per.minute = cbind(userid, steps.per.minute)
  colnames(steps.per.minute) = c("userid", "date", "fitbit.minutes.worn")
  return(steps.per.minute)
}


bouts.of.activity = function(userid) {
  ## step counts per minute
  path = paste0(csv.folder, "kpwhri.", aim, ".", userid, ".fitbit-data-per-minute.csv")
  steps.per.minute = read.csv(path, header = T)
  ## some users have rows with time like "17:24:S"
  steps.per.minute$time = gsub("S", "00", steps.per.minute$time)
  if (grepl("/", steps.per.minute$date[1])) {
    ## for dates like "23/06/2019" (user 10006)
    steps.per.minute$date = dmy(steps.per.minute$date)
  } else {
    ## for dates like "2019-06-28" (user 10008)
    steps.per.minute$date = ymd(steps.per.minute$date)
  }
  steps.per.minute$time = ymd_hms(paste(
    steps.per.minute$date, steps.per.minute$time, sep = " "
  ))

  ## decision times
  path = paste0(csv.folder, "kpwhri.", aim, ".", userid, ".walking-suggestion-decisions.csv")
  suggestions = read.csv(path, header = T)
  if (!"Decision.Date" %in% colnames(suggestions)) {
    ## date and time are saved in the same column
    ## for users 10044 and 10161, Decision.Time is like "2019-11-23 18:23 -0800" 
    suggestions$Decision.Time = ymd_hm(substr(suggestions$Decision.Time, 1, 16))
    suggestions$Decision.Date = date(suggestions$Decision.Time)
  } else {
    ## date and time are saved in separate columns
    suggestions$Decision.Time = ymd_hms(paste(
      suggestions$Decision.Date, suggestions$Decision.Time, sep = " "
    ))
    suggestions$Decision.Date = ymd(suggestions$Decision.Date)
  }

  ## initialize output matrix
  date.min = min(steps.per.minute$date)
  date.max = max(steps.per.minute$date)
  dates = seq(date.min, date.max, "days")
  nday = as.numeric(date.max - date.min) + 1
  user.bouts = data.frame(userid, date = dates, unprompted.bouts = rep(0, nday))
  
  for (d in 1:nday) {
    above.threshold = subset(steps.per.minute, date == user.bouts$date[d])
    row.names(above.threshold) = NULL ## reset row index
    above.threshold$steps = as.numeric(above.threshold$steps >= 40)
    above.threshold$steps[is.na(above.threshold$steps)] = 0
    n.minute = nrow(above.threshold)
    
    ## decision times today
    prompted.times = subset(
      suggestions, Decision.Date == user.bouts$date[d] & treated == 1
    )
    prompted.times = prompted.times$Decision.Time
    
    running.sum = 0
    is.bout = 0
    i = 1
    j = 1
    tmp = c()
    while (i <= n.minute & j <= n.minute) {
      ## let i be the next 1
      while (i <= n.minute & above.threshold$steps[i] == 0) {
        i = i + 1
      }
      ## terminate if the dataset has be traversed
      if (i > n.minute) {
        break
      }
      ## start from the interval with j = i
      j = i
      running.sum = above.threshold$steps[j]
      ## find the interval with length 10
      while (j <= n.minute & j - i < 9) {
        j = j + 1
        running.sum = running.sum + above.threshold$steps[j]
      }
      ## if this is a bout, find the end of the bout
      while (j <= n.minute & running.sum / (j - i + 1) >= 0.8 & sum(above.threshold$steps[(j-5):j]) > 0) { ## 6 consecutive numbers
        is.bout = 1
        j = j + 1
        running.sum = running.sum + above.threshold$steps[j]
      }
      tmp = rbind(tmp, c(i, j, running.sum))
      ## if this is a bout
      if (is.bout == 1) {
        ## check if this is a prompted bout
        start.time = above.threshold$time[i]
        flag.prompted = 0
        if (length(prompted.times) > 0){
          for (k in 1:length(prompted.times)) {
            if (start.time %within% interval(prompted.times[k], prompted.times[k] + dminutes(30))) {
              flag.prompted = 1
              break
            }
          }
        }
        ## if this is un unprompted bout, add 1 to the counter
        if (flag.prompted == 0){
          user.bouts[d, "unprompted.bouts"] = user.bouts[d, "unprompted.bouts"] + 1
        } else {
          # print(user.bouts$date[d])
        }
        ## reset
        is.bout = 0
        ## start from the next minute
        i = j + 1
      } else {
        ## if this is not a bout, move i to the next minute
        i = i + 1
      }
      ## terminate if the dataset has be traversed
      if (j > n.minute) {
        break
      }
    }
  }
  return(user.bouts)
}


for (aim in aims){
  filtered.userid = intersect(userid.by.aim[[aim]], IDtable$user)
  
  dat = list()
  j = 1
  
  ## use third processing data
  ## however, third processing data does not contain the 7 days before study
  load(paste0(third.proc.folder, "improved-step-related.RData"))
  for (k in 1:length(data.all)) {
    ## re-imputed step counts in third processing data
    userid = data.all[[k]]$daily$user[1]
    if (is.null(userid)) next
    if (!(userid %in% filtered.userid)) next

    dat.user1 = data.all[[k]]$step_features
    colnames(dat.user1)[1:2] = c("day", "decision.time")
    
    if (userid == 10161) { ## k = 33
      ## there are 5 more rows in dat.user1 than in dat.user2, corresponding to day 77
      ## the rewards are all missing in the 5 decision stages
      dat.user1 = dat.user1[-c(381:385), ]
    }
    
    ## add space to record information before study begins
    before_study = data.frame(matrix(NA, nrow = 7*5, ncol = ncol(dat.user1)))
    colnames(before_study) = colnames(dat.user1)
    before_study$day = rep(c(-7:-1), each = 5)
    before_study$decision.time = rep(c(1:5), 7)
    dat.user1 = rbind(dat.user1, before_study)
    dat.user1 = dat.user1 %>% arrange(day, decision.time)

    ## features in second processing data
    load(paste0(processed.data.folder, userid, ".Rdata"))
    dat.user2 = subset(tmp_df, select = c(
      day, decision.time, availability, probability, action,
      dosage, engagement, other.location, temperature
    ))

    ## match day to date
    user.day = day.to.date(userid)
    # user.day = subset(data.all[[k]]$daily, select = c(user, studyDay, date))
    # colnames(user.day) = c("userid", "day", "date")
    # user.day$date = ymd(user.day$date)
    ## find the total app view per day
    user.app = app.view(userid)
    ## calculate bouts of physical activity
    user.bouts = bouts.of.activity(userid)
    ## whether the Fitbit is worn on each day
    user.fitbit.minutes = fitbit.minutes.worn(userid)
    ## combine date and unprompted bouts
    extra_info = Reduce(
      function(x, y) merge(x, y, by = c("userid", "date"), all=TRUE), 
      list(user.day, user.app, user.bouts, user.fitbit.minutes)
    )
    extra_info$fitbit.worn = (extra_info$fitbit.minutes.worn >= 480)
    extra_info$app.view[is.na(extra_info$app.view)] = 0  ## no record means no app view
    extra_info = extra_info[!is.na(extra_info$day),]
    
    dat.user = merge(dat.user1, dat.user2, by = c("day", "decision.time"), all = TRUE)
    dat.user$userid = userid
    dat.user = merge(x = dat.user, y = extra_info, by = c("userid", "day"), all = TRUE)
    dat.user = dat.user %>% arrange(day, decision.time)
    dat[[j]] = dat.user
    print(j)
    j = j + 1
  }
  
  dat = do.call(rbind, dat)
  dat = dat %>% arrange(userid, day, decision.time)
  # dat$day = dat$day - 1
  write.csv(dat, paste0(data.folder, aim, ".csv"), row.names = FALSE)
}

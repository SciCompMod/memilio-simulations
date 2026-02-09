#!/usr/bin/env Rscript
# Benchmark R version of the SEIR metapop model using deSolve::ode (RK4)

if (!requireNamespace("deSolve", quietly = TRUE)) {
  stop("Please install package 'deSolve' first: install.packages('deSolve')")
}

args <- commandArgs(trailingOnly = TRUE)
region_grid <- c(1, 2, 4, 8, 16, 32, 64, 128, 256)
dt <- 0.1
tmax <- 500
n_runs <- 40
use_naive_linalg <- FALSE

if (length(args) > 0) {
  i <- 1
  remaining <- c()
  while (i <= length(args)) {
    a <- args[[i]]
    if (grepl("^--dt=", a)) {
      dt <- as.numeric(sub("^--dt=", "", a))
    } else if (a == "--dt" && i < length(args)) {
      i <- i + 1
      dt <- as.numeric(args[[i]])
    } else if (grepl("^--tmax=", a)) {
      tmax <- as.numeric(sub("^--tmax=", "", a))
    } else if (a == "--tmax" && i < length(args)) {
      i <- i + 1
      tmax <- as.numeric(args[[i]])
    } else if (grepl("^--runs=", a)) {
      runs_candidate <- as.integer(sub("^--runs=", "", a))
      if (!is.na(runs_candidate) && runs_candidate > 0) {
        n_runs <- runs_candidate
      }
    } else if (a == "--runs" && i < length(args)) {
      i <- i + 1
      runs_candidate <- as.integer(args[[i]])
      if (!is.na(runs_candidate) && runs_candidate > 0) {
        n_runs <- runs_candidate
      }
    } else if (grepl("^--use-naive=", a)) {
      val <- tolower(sub("^--use-naive=", "", a))
      use_naive_linalg <- val %in% c("1","true","t","yes","y")
    } else if (a == "--use-naive" && i < length(args)) {
      i <- i + 1
      val <- tolower(args[[i]])
      use_naive_linalg <- val %in% c("1","true","t","yes","y")
    } else if (a == "--naive") {
      use_naive_linalg <- TRUE
    } else if (a == "--fast") {
      use_naive_linalg <- FALSE
    } else {
      remaining <- c(remaining, a)
    }
    i <- i + 1
  }
  if (length(remaining) > 0) {
    suppressWarnings(region_grid <- as.integer(remaining))
    region_grid <- region_grid[!is.na(region_grid)]
    if (length(region_grid) == 0) {
      region_grid <- c(1, 2, 4, 8, 16, 32, 64, 128, 256)
    }
  }
}
n_runs <- max(1L, as.integer(n_runs))
use_naive_linalg <- isTRUE(use_naive_linalg)
cat("RK4 benchmark uses ", if (use_naive_linalg) "naive" else "fast", " matrix-vector ops.\n", sep = "")

default_pop <- 10000
contact_rate <- 2.7
transmission_prob <- 0.07333
T_E <- 3.335
T_I <- 8.097612257
coeffStoE <- 0.5 * contact_rate * transmission_prob

naive_matvec <- function(A, x) {
  n <- nrow(A); m <- ncol(A)
  if (length(x) != m) stop("Dimension mismatch in naive_matvec")
  y <- numeric(n)
  for (i in 1:n) {
    s <- 0.0
    ai <- A[i, ]
    for (j in 1:m) {
      s <- s + ai[j] * x[j]
    }
    y[i] <- s
  }
  y
}

naive_t_matvec <- function(A, x) {
  n <- nrow(A); m <- ncol(A)
  if (length(x) != n) stop("Dimension mismatch in naive_t_matvec")
  y <- numeric(m)
  for (j in 1:m) {
    s <- 0.0
    for (i in 1:n) {
      s <- s + A[i, j] * x[i]
    }
    y[j] <- s
  }
  y
}

rhs_factory <- function(n_regions, commuting_matrix, pop_after_commuting, use_naive){
  matvec <- if (use_naive) {
    naive_matvec
  } else {
    function(A, x) as.numeric(A %*% x)
  }
  t_matvec <- if (use_naive) {
    naive_t_matvec
  } else {
    function(A, x) as.numeric(t(A) %*% x)
  }
  function(t, y, pars){
    S <- y[seq(1, 4*n_regions, by=4)]
    E <- y[seq(2, 4*n_regions, by=4)]
    I <- y[seq(3, 4*n_regions, by=4)]
    R <- y[seq(4, 4*n_regions, by=4)]
    N <- S + E + I + R
    infected_pop <- I
    tmp <- t_matvec(commuting_matrix, infected_pop)
    infectious_share <- tmp / pop_after_commuting
    infections_due_commuting <- matvec(commuting_matrix, infectious_share)
    lambda <- ( (I / N) + infections_due_commuting) * coeffStoE * S
    dS <- -lambda
    dE <- lambda - E / T_E
    dI <- E / T_E - I / T_I
    dR <- I / T_I
    list(as.numeric(rbind(dS,dE,dI,dR)))
  }
}

simulate_once <- function(n_regions, tmax=500, dt=0.1, use_naive=FALSE) {
  t_total_start <- proc.time()
  S <- rep(default_pop, n_regions)
  E <- rep(0, n_regions); E[1] <- 100; S[1] <- S[1] - 100
  I <- rep(0, n_regions)
  R <- rep(0, n_regions)
  Cmat <- diag(n_regions)
  pop_after <- rep(default_pop, n_regions)
  y0 <- as.numeric(rbind(S,E,I,R))
  times <- seq(0, tmax, by=dt)
  rhs <- rhs_factory(n_regions, Cmat, pop_after, use_naive)
  t_sim_start <- proc.time()
  sol <- deSolve::ode(y = y0, times = times, func = rhs, parms = NULL, method = "rk4")
  t_sim_end <- proc.time()
  t_total_end <- proc.time()
  runtime <- (t_sim_end - t_sim_start)["elapsed"]
  total_no_io <- (t_total_end - t_total_start)["elapsed"]
  list(steps = nrow(sol), runtime = as.numeric(runtime), total_no_io = as.numeric(total_no_io))
}

benchmark_region <- function(n_regions, runs, tmax, dt, use_naive) {
  cat("[RK4] Regions=", n_regions, " ... ", sep = "")
  flush.console()
  sim_durations <- numeric(runs)
  total_durations <- numeric(runs)
  steps <- NA_integer_
  for (k in seq_len(runs)) {
    res <- simulate_once(n_regions = n_regions, tmax = tmax, dt = dt, use_naive = use_naive)
    steps <- res$steps
    sim_durations[k] <- res$runtime
    total_durations[k] <- res$total_no_io
  }
  result <- list(
    steps = steps,
    runtime = median(sim_durations),
    total_no_io = median(total_durations)
  )
  cat("done.\n")
  result
}

out_path <- "/localdata1/code_2025/memilio/compare_results/r_benchmark_rk4.csv"
res <- lapply(
  region_grid,
  function(n) benchmark_region(n_regions = n, runs = n_runs, tmax = tmax, dt = dt, use_naive = use_naive_linalg)
)
steps <- sapply(res, `[[`, "steps")
runtime <- sapply(res, `[[`, "runtime")
total_no_io <- sapply(res, `[[`, "total_no_io")
bench <- data.frame(Regions = region_grid, TimeSteps = steps, RuntimeSeconds = runtime, TotalNoIOSeconds = total_no_io)
write.csv(bench, out_path, row.names = FALSE, quote = FALSE)
cat("R RK4 benchmark written to ", out_path, " (dt=", dt, ", tmax=", tmax, ", runs=", n_runs, ")\n", sep="")
print(bench)

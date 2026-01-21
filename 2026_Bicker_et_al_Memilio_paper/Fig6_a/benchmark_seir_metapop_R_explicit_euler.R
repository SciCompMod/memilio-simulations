#!/usr/bin/env Rscript
# Benchmark R Version of SEIR Metapop Model Scaling over Regions

args <- commandArgs(trailingOnly = TRUE)
# Defaults
region_grid <- c(1, 2, 4, 8, 16, 32, 64, 128, 256)
dt <- 0.1
tmax <- 500
n_runs <- 40

# Simple CLI parsing for --dt and --tmax; remaining numeric args become regions
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

contact_rate <- 2.7
transmission_prob <- 0.07333
T_E <- 3.335
T_I <- 8.097612257
coeffStoE <- 0.5 * contact_rate * transmission_prob

# Pure R helper functions for matrix-vector products (without %*% / BLAS)
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
  # y = t(A) %*% x, without constructing t()
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

rhs_factory <- function(n_regions, commuting_matrix, pop_after_commuting){
  function(t, y, pars){
    # layout: for region r: S,E,I,R contiguous -> total length 4*n_regions
    S <- y[seq(1, 4*n_regions, by=4)]
    E <- y[seq(2, 4*n_regions, by=4)]
    I <- y[seq(3, 4*n_regions, by=4)]
    R <- y[seq(4, 4*n_regions, by=4)]
    N <- S + E + I + R
    # compute infectious_share_per_region purely in R (without %*%)
    infected_pop <- I
    # tmp := t(C) %*% infected_pop
    tmp <- naive_t_matvec(commuting_matrix, infected_pop)
    infectious_share <- tmp / pop_after_commuting
    # infections_due_commuting := C %*% infectious_share
    infections_due_commuting <- naive_matvec(commuting_matrix, infectious_share)
    lambda <- ( (I / N) + infections_due_commuting) * coeffStoE * S
    dS <- -lambda
    dE <- lambda - E / T_E
    dI <- E / T_E - I / T_I
    dR <- I / T_I
    list(as.numeric(rbind(dS,dE,dI,dR)))
  }
}

# Naive explicit Euler integrator for ODEs with constant step
# y0: numeric vector of initial values
# times: equidistant time points (dt = times[i+1]-times[i])
# rhs: function (t, y, pars) -> list(dy)
# parms: optional parameters (not used, placeholder for compatibility)
euler_integrate <- function(y0, times, rhs, parms=NULL) {
  n <- length(times)
  if (n < 2) stop("times must have length >= 2")
  dt <- diff(times)
  # Constant step size required
  if (any(abs(dt - dt[1]) > 1e-12)) stop("Non-uniform time steps are not supported by naive Euler.")
  h <- dt[1]
  y <- as.numeric(y0)
  m <- length(y)
  out <- matrix(NA_real_, nrow=n, ncol=m+1)
  colnames(out) <- c("time", paste0("V", seq_len(m)))
  out[1,] <- c(times[1], y)
  for (i in 1:(n-1)) {
    t_i <- times[i]
    dy <- rhs(t_i, y, parms)[[1]]
    # Euler step
    y <- y + h * dy
    out[i+1,] <- c(times[i+1], y)
  }
  out
}

simulate_once <- function(n_regions, tmax=500, dt=0.1){
  # Start total timer as early as possible (include setup, exclude file IO outside)
  t_total_start <- proc.time()
  # initial populations
  S <- rep(10000, n_regions)
  E <- rep(0, n_regions); E[1] <- 100; S[1] <- S[1] - 100
  I <- rep(0, n_regions)
  R <- rep(0, n_regions)
  # commuting identity
  Cmat <- diag(n_regions)
  # population after commuting (stays same)
  pop_after <- rep(10000, n_regions)
  y0 <- as.numeric(rbind(S,E,I,R))
  times <- seq(0, tmax, by=dt)
  rhs <- rhs_factory(n_regions, Cmat, pop_after)
  t_sim_start <- proc.time()
  sol <- euler_integrate(y0=y0, times=times, rhs=rhs, parms=NULL)
  t_sim_end <- proc.time()
  t_total_end <- proc.time()
  runtime <- (t_sim_end - t_sim_start)["elapsed"]
  total_no_io <- (t_total_end - t_total_start)["elapsed"]
  list(steps=nrow(sol), runtime=as.numeric(runtime), total_no_io=as.numeric(total_no_io))
}

benchmark_region <- function(n_regions, runs, tmax, dt){
  sim_durations <- numeric(runs)
  total_durations <- numeric(runs)
  steps <- NA_integer_
  for (k in seq_len(runs)) {
    res <- simulate_once(n_regions = n_regions, tmax = tmax, dt = dt)
    steps <- res$steps
    sim_durations[k] <- res$runtime
    total_durations[k] <- res$total_no_io
  }
  list(
    steps = steps,
    runtime = median(sim_durations),
    total_no_io = median(total_durations)
  )
}

out_path <- "compare_results/r_benchmark_euler.csv"
res <- lapply(region_grid, function(n) benchmark_region(n_regions = n, runs = n_runs, tmax = tmax, dt = dt))
steps <- sapply(res, `[[`, "steps")
runtime <- sapply(res, `[[`, "runtime")
total_no_io <- sapply(res, `[[`, "total_no_io")
bench <- data.frame(Regions=region_grid, TimeSteps=steps, RuntimeSeconds=runtime, TotalNoIOSeconds=total_no_io)
write.csv(bench, out_path, row.names = FALSE, quote = FALSE)
cat("R benchmark written to", out_path, "(dt=", dt, ", tmax=", tmax, ", runs=", n_runs, ")\n", sep="")
print(bench)

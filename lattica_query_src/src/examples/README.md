### How to run an example:
- If you want a local run set `RUN_LOCAL=true`, otherwise set it to `false` and set the `beUrl` to point to `app` of the correct environment.
- Make sure you configure the `jwt` token that corresponds to the model you want to use.
- In `tsup.config.ts` append `src/examples/<eg>.ts` to `entry` list (after `src/index.ts`).
- In `RunExample.html`, change the script path to the example you want to run `dist/examples/<eg>.mjs`.
- Spawn an local web server in the root directory of the project `python3 http_server.py`.
- Run the `RunExample.html` file in the browser: `http://localhost:3010/RunExample.html`.
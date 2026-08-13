# ==== DEMO BEZ WIDGETA — pozivaj funkcije umjesto dugmadi ====
SCENE_IDX = 3

state = {}
def reset(idx=None):
    global SCENE_IDX
    if idx is not None: SCENE_IDX = idx
    lat = retrieve_row_from_lmdb(env,'latents',np.float16,SCENE_IDX,shape=LAT_SHAPE[1:]).astype(np.float32)
    real = torch.from_numpy(lat).to(device=device, dtype=torch.bfloat16).unsqueeze(0)
    state['context'] = real[:,:N_CTX].clone()
    state['latents'] = [real[:,:N_CTX].clone()]
    state['prev'] = 'still'
    state['history'] = []
    show('start (pravi kontekst)')

def show(msg=''):
    frames = decode(torch.cat(state['latents'], dim=1))
    print(f"{msg}   niz: {' -> '.join(state['history']) or '(prazno)'}")
    print(f"{len(frames)} frejmova  (prvih {1+4*(N_CTX-1)} = pravi kontekst, ostalo generisano)")
    display(strip(frames))

def go(*actions):
    """go('up')  |  go('up','down')  |  go('right','right','left')"""
    for a in actions:
        assert a in DIRS, f"nepoznata akcija {a}; izaberi: {list(DIRS)}"
        t0 = time.time()
        blk = step_block(state['context'], state['prev'], a)
        state['latents'].append(blk); state['context'] = blk
        state['prev'] = a; state['history'].append(a)
        print(f"  {a}  [{time.time()-t0:.1f}s]")
    show()

def save(name=None):
    import imageio
    frames = decode(torch.cat(state['latents'], dim=1))
    name = name or ('_'.join(state['history']) or 'ctx')
    p = f'/home/mls10/logs/interactive_{name}.mp4'
    imageio.mimsave(p, list(frames), fps=4, macro_block_size=1)
    print('snimljeno:', p)

reset()
print("\nkoristi:  go('up')   go('up','down')   go('right','right')   reset()   save()")

"""Bounded official-source downloads, including selective ZIP reads over HTTP Range."""
import io
import json
import re
import shutil
import time
import urllib.request
from pathlib import Path, PurePosixPath


def safe_relative(value):
    p=PurePosixPath(value)
    if p.is_absolute() or '\\' in value or ':' in value or '..' in p.parts or not p.parts:
        raise ValueError('unsafe relative path')
    return p


def free_space(path, required=0, reserve=12*1024**3):
    p=Path(path)
    while not p.exists(): p=p.parent
    if shutil.disk_usage(p).free < required+reserve:
        raise ValueError('disk reserve would be violated')


class RangeReader(io.RawIOBase):
    """No full-body fallback. Limits transferred bytes, including ZIP directory."""
    def __init__(self,url,max_bytes=2*1024**3,opener=urllib.request.urlopen):
        self.url,self.max_bytes,self.opener=url,max_bytes,opener
        self.pos,self.transferred,self.size=0,0,None
        self.window_start,self.window=0,b''
        self._fetch(0,0)

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        pos=offset+(0 if whence==0 else self.pos if whence==1 else self.size if whence==2 else -10**30)
        if pos<0: raise ValueError('invalid seek')
        self.pos=pos
        return pos

    def _fetch(self,start,end):
        n=end-start+1
        if self.transferred+n>self.max_bytes: raise ValueError('transfer budget exceeded')
        req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
        for attempt in range(4):
            try:
                with self.opener(req,timeout=60) as r:
                    if r.status!=206: raise ValueError('server must honor Range with HTTP 206')
                    m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',r.headers.get('Content-Range',''))
                    if not m or tuple(map(int,m.groups()[:2]))!=(start,end):
                        raise ValueError('unexpected Content-Range')
                    size=int(m.group(3))
                    if self.size is not None and size!=self.size: raise ValueError('remote size changed')
                    self.size=size
                    self.transferred+=n
                    data=r.read(n+1)
                    if len(data)!=n: raise ValueError('short or oversized range response')
                    return data
            except (OSError,TimeoutError):
                if attempt==3: raise
                if self.transferred+n>self.max_bytes: raise ValueError('transfer budget exceeded')
                time.sleep(2**attempt)

    def read(self,n=-1):
        n=max(0,self.size-self.pos) if n is None or n<0 else min(n,max(0,self.size-self.pos))
        offset=self.pos-self.window_start
        if 0<=offset and offset+n<=len(self.window):
            data=self.window[offset:offset+n]; self.pos+=n
            return data
        parts=[]
        while n:
            count=min(n,4*1024**2)
            parts.append(self._fetch(self.pos,self.pos+count-1)); self.pos+=count; n-=count
        return b''.join(parts)

    def prefetch(self,start,length):
        if length>32*1024**2: raise ValueError('prefetch exceeds memory cap')
        position=self.pos
        self.seek(start); data=self.read(length); self.seek(position)
        self.window_start,self.window=start,data


def download(url,path,max_bytes=128*1024**2):
    """Atomic small-file download, bounded even when Content-Length is missing."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    side=path.with_name(path.name+'.download.json')
    if path.exists() and side.exists():
        meta=json.loads(side.read_text())
        from .io import sha256
        if meta['url']!=url or meta['sha256']!=sha256(path): raise ValueError('existing download provenance mismatch')
        return path
    if path.exists(): raise ValueError('unverified existing file; choose a new output directory')
    part=path.with_name(path.name+'.part')
    for attempt in range(4):
        try:
            free_space(path,max_bytes)
            with urllib.request.urlopen(url,timeout=60) as r, part.open('wb') as f:
                count=0
                while True:
                    block=r.read(min(1024**2,max_bytes-count+1))
                    if not block: break
                    count+=len(block)
                    if count>max_bytes: raise ValueError('file download exceeds size cap')
                    free_space(path,len(block)); f.write(block)
            from .io import sha256,write_json
            checksum=sha256(part); part.replace(path)
            write_json(side,dict(url=url,bytes=count,sha256=checksum))
            return path
        except (OSError,TimeoutError):
            if attempt==3: raise
            time.sleep(2**attempt)


def extract_member(archive,name,root,max_bytes=128*1024**2):
    rel=safe_relative(name); info=archive.getinfo(name)
    if info.file_size>max_bytes: raise ValueError('ZIP entry exceeds cap')
    if (info.external_attr>>16)&0o170000==0o120000: raise ValueError('ZIP symlink forbidden')
    root=Path(root).resolve(); path=root.joinpath(*rel.parts)
    if not path.resolve().is_relative_to(root): raise ValueError('ZIP destination escapes root')
    path.parent.mkdir(parents=True,exist_ok=True); free_space(path,info.file_size)
    if path.exists():
        import zlib
        if path.stat().st_size==info.file_size and zlib.crc32(path.read_bytes())==info.CRC: return path
        raise ValueError('existing ZIP member checksum mismatch')
    part=path.with_name(path.name+'.part')
    if isinstance(archive.fp,RangeReader):
        # A ZIP member otherwise needs separate remote header/name/payload reads.
        # Extra 4KiB covers differing local header extras; overflow still falls
        # back to strict, budget-accounted Range reads.
        length=min(archive.fp.size-info.header_offset,
                   30+len(info.filename.encode('utf-8'))+len(info.extra)+info.compress_size+4096)
        archive.fp.prefetch(info.header_offset,length)
    with archive.open(info) as source, part.open('wb') as target:
        shutil.copyfileobj(source,target,1024**2) # ZipFile enforces CRC and declared length.
    if part.stat().st_size!=info.file_size: raise ValueError('ZIP extracted size mismatch')
    part.replace(path)
    return path
